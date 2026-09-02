#[cfg(target_os = "macos")]
use std::cmp::min;

use cubecl::{
    Runtime,
    client::ComputeClient,
    features::MmaConfig,
    ir::{ElemType, VectorSize},
};
use cubek_std::{
    MatmulProblemSize, MatrixLayout,
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
};

use crate::{
    definition::{
        MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes,
    },
    multi_level::{
        PartitionSize, StageSize, TileSize,
        components::{
            global::{InputLoadFlow, LoadFlows},
            stage::PartitionBuffering,
            tile::TileMatmulKind,
        },
        definition::{
            BatchMatmulBlueprint, MultiRowStrategy, SwizzleModes, TilingScheme, adjust_dtypes,
        },
        routines::selector::is_tiny,
        stage::SwizzleMode,
    },
};

pub const NUM_SM_APPROX: u32 = 50;
pub const NUM_TENSOR_CORES_APPROX: u32 = 4;

#[derive(Debug)]
/// Options to select the best plane matmul [selection](BatchMatmulBlueprint).
pub struct PlaneTilingBlueprintOptions {
    pub partition_k: Option<u32>,
    pub specialized: bool,
    pub swizzled: bool,
    pub row_count: Option<u32>,
    pub multi_row_strategy: MultiRowStrategy,
    pub partition_buffering: Option<PartitionBuffering>,
    /// Enables the tiny selector when the [matmul problem](MatmulProblem) is flagged as tiny.
    pub tiny_selection_enabled: bool,
    /// K-stages the routine's k-loop consumes per iteration (`NumStages::stage_buffering`).
    pub stage_buffering: u32,
}

impl Default for PlaneTilingBlueprintOptions {
    fn default() -> Self {
        Self {
            partition_k: None,
            specialized: false,
            swizzled: false,
            row_count: None,
            multi_row_strategy: MultiRowStrategy::default(),
            partition_buffering: None,
            tiny_selection_enabled: false,
            stage_buffering: 1,
        }
    }
}

pub fn infer_blueprint_plane<R: Runtime>(
    tile_matmul: TileMatmulKind,
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    vector_sizes: &MatmulVectorSizes,
    options: PlaneTilingBlueprintOptions,
) -> Result<(BatchMatmulBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, tile_matmul.requires_accelerator());

    if plane_dim == 1 {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim: 1 },
        ));
    }

    let tile_size = find_instruction_size::<R, _, _>(
        client,
        (
            dtypes.lhs_register,
            dtypes.rhs_register,
            dtypes.acc_register,
        ),
        (problem.m, problem.n, problem.k).into(),
        (None, None, None),
        |c, cfg| tile_matmul.is_supported(c, cfg),
        |c, l, r, a| tile_matmul.supported_sizes(c, l, r, a),
    )?;

    if options.tiny_selection_enabled && is_tiny(problem, &tile_size) {
        return Ok((
            selection_tiny(client, problem, tile_size, plane_dim, tile_matmul),
            dtypes,
        ));
    }

    let row_count = options.row_count.unwrap_or_else(|| {
        #[cfg(target_os = "macos")]
        // If we allow too many units it will select a large plane_count and fail with Cube Dim too large
        let max_units_per_cube = min(client.properties().hardware.max_units_per_cube, 256);
        #[cfg(not(target_os = "macos"))]
        let max_units_per_cube = client.properties().hardware.max_units_per_cube;

        let max_plane_per_cube = max_units_per_cube / plane_dim;
        // Compensate for register use
        let precision_factor = match dtypes.lhs_stage.size() >= 4 {
            true => 2,
            false => 1,
        };
        let mut tile_factor = tile_size.n().div_ceil(4);
        if problem.m as u32 <= tile_size.m() * 4 || problem.n as u32 <= tile_size.n() * 4 {
            tile_factor = 8;
        }
        max_plane_per_cube / (tile_factor * precision_factor)
    });

    if row_count == 0 {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim },
        ));
    }

    let max_stage_tiles_m = if options.swizzled && problem.lhs_layout == MatrixLayout::ColMajor {
        max_swizzle_tiles(tile_size.m() as usize, dtypes.lhs_stage.size())
    } else {
        usize::MAX
    };
    let max_partition_shape_n = if options.swizzled && problem.rhs_layout == MatrixLayout::RowMajor
    {
        max_swizzle_tiles(tile_size.n() as usize, dtypes.rhs_stage.size())
    } else {
        usize::MAX
    };

    let (rows_per_plane, stage_size_m, partition_shape_n) = select_size(
        options.multi_row_strategy,
        row_count as usize,
        tile_size.m() as usize,
        tile_size.n() as usize,
        problem.m,
        problem.n,
        vector_sizes.lhs,
        vector_sizes.rhs,
        max_stage_tiles_m,
        max_partition_shape_n,
    );

    let mut partition_shape_k = options
        .partition_k
        .unwrap_or_else(|| plane_dim / tile_size.k());

    if options.swizzled {
        let max_swizzle_span = SwizzleMode::B128.span_size() as u32;
        if problem.lhs_layout == MatrixLayout::RowMajor {
            let elem_size = dtypes.lhs_global.size() as u32;
            while partition_shape_k * tile_size.k() * elem_size > max_swizzle_span {
                partition_shape_k /= 2;
            }
        }
        if problem.rhs_layout == MatrixLayout::ColMajor {
            let elem_size = dtypes.rhs_global.size() as u32;
            while partition_shape_k * tile_size.k() * elem_size > max_swizzle_span {
                partition_shape_k /= 2;
            }
        }
    }

    let tiles_per_partition = PartitionSize::new(
        rows_per_plane as u32,
        partition_shape_n as u32,
        partition_shape_k,
    );

    let partitions_per_stage = StageSize::new(stage_size_m as u32, 1, 1);

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(tiles_per_partition)
        .with_stage_size(partitions_per_stage)
        .build()
        .unwrap();

    let partition_buffering = options.partition_buffering.unwrap_or_else(|| {
        if tiling_scheme.tiles_per_stage_partition_along_n() > 1 {
            PartitionBuffering::Double
        } else {
            PartitionBuffering::Single
        }
    });

    let cube_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountStrategy::FromProblem,
    };

    let hypercube = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::SwizzleRow(4))
        .cube_count_strategy(cube_count_strategy)
        .build();

    let mut builder = BatchMatmulBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
        .partition_buffering(partition_buffering)
        .stage_buffering(options.stage_buffering)
        .hypercube_blueprint(hypercube);

    if options.specialized {
        builder = builder.load_specialization_config(LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        });
    }

    if options.swizzled {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m() as usize,
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k() as usize,
        };

        let lhs = select_swizzle(lhs_swizzle_dim, dtypes.lhs_stage, vector_sizes.lhs);
        let rhs = select_swizzle(rhs_swizzle_dim, dtypes.rhs_stage, vector_sizes.rhs);
        builder = builder.shared_swizzle(SwizzleModes {
            lhs,
            rhs,
            ..Default::default()
        });
    }

    Ok((builder.build(), dtypes))
}

/// All modes currently use atom size 16
const SWIZZLE_ATOM: usize = 16;

fn max_swizzle_tiles(instruction_size: usize, elem_size: usize) -> usize {
    (SwizzleMode::B128.span_size() / (instruction_size * elem_size)).max(1)
}

pub fn select_swizzle(swizzle_dim: usize, elem: ElemType, vector_size: VectorSize) -> SwizzleMode {
    // Vector size exceeds swizzle atom
    if elem.size() * vector_size > SWIZZLE_ATOM {
        return SwizzleMode::None;
    }
    let swizzle_dim_bytes = swizzle_dim * elem.size();
    if !swizzle_dim_bytes.is_power_of_two() {
        return SwizzleMode::None;
    }
    match swizzle_dim_bytes {
        32 => SwizzleMode::B32,
        64 => SwizzleMode::B64,
        128 => SwizzleMode::B128,
        _ => SwizzleMode::None,
    }
}

#[allow(clippy::too_many_arguments)]
fn select_size(
    strategy: MultiRowStrategy,
    plane_count: usize,
    instruction_m: usize,
    instruction_n: usize,
    problem_m: usize,
    problem_n: usize,
    lhs_vector_size: VectorSize,
    rhs_vector_size: VectorSize,
    max_stage_tiles_m: usize,
    max_partition_shape_n: usize,
) -> (usize, usize, usize) {
    let rows = match strategy {
        MultiRowStrategy::Never => 1,
        MultiRowStrategy::Always(count) => count,
        MultiRowStrategy::Adaptive {
            minimum_stage_count,
        } => {
            if problem_m > plane_count * instruction_m * minimum_stage_count as usize {
                2
            } else {
                1
            }
        }
    } as usize;

    // The number of rows handled per plane cannot exceed the number of available
    // planes: otherwise `plane_count / rows` underflows to 0, producing a degenerate
    // tiling scheme with `stage_size.m == 0` that divides by zero in
    // `BatchMatmulBlueprint::cube_launch_info`. Clamp so there is always at least one stage
    // along `m` (e.g. a large `problem_m` requesting 2 rows when only 1 plane fits).
    let rows = rows.min(plane_count).max(1);

    // For narrow outputs, cover the useful N tiles with a bounded power-of-two partition.
    // Otherwise preserve the existing geometry unless swizzling requires a smaller span.
    let required_n_tiles = problem_n.div_ceil(instruction_n).max(1);
    let balance_stage_loads = required_n_tiles < plane_count;
    let partition_shape_n_limit = plane_count.min(max_partition_shape_n).max(1);
    let partition_shape_n = if balance_stage_loads {
        power_of_two_at_most(required_n_tiles, partition_shape_n_limit)
    } else if partition_shape_n_limit < plane_count {
        power_of_two_at_most(plane_count, partition_shape_n_limit)
    } else {
        plane_count
    };

    // Balance vector loads between the two stages. K cancels from this relationship:
    //
    // stage_m * instruction_m * rows / lhs_vector_size
    //     ~= instruction_n * partition_n / rhs_vector_size
    let stage_m_numerator = instruction_n * partition_shape_n * lhs_vector_size;
    let stage_m_denominator = instruction_m * rows * rhs_vector_size;
    let default_stage_size_m = plane_count / rows;
    let desired_stage_size_m = if balance_stage_loads {
        stage_m_numerator.div_ceil(stage_m_denominator)
    } else {
        default_stage_size_m
    };

    // Adjusted geometries use a power-of-two stage size no larger than the N partition,
    // preserving the K factor needed by tilewise readers.
    let max_swizzled_stage_size_m = (max_stage_tiles_m / rows).max(1);
    let stage_size_m_limit = default_stage_size_m
        .min(max_swizzled_stage_size_m)
        .min(partition_shape_n)
        .max(1);
    let stage_size_m = if !balance_stage_loads && desired_stage_size_m <= stage_size_m_limit {
        desired_stage_size_m
    } else {
        power_of_two_at_most(desired_stage_size_m, stage_size_m_limit)
    };

    (rows, stage_size_m, partition_shape_n)
}

fn power_of_two_at_most(target: usize, limit: usize) -> usize {
    let limit = limit.max(1);
    let target = target.clamp(1, limit);
    let rounded = target.next_power_of_two();

    if rounded <= limit {
        rounded
    } else {
        1 << limit.ilog2()
    }
}

/// The instruction shape for this problem: [`crate::multi_level::find_instruction_size`]
/// with matmul's own error on the empty case, and the client and element triple
/// bound into its capability closures. The stages itself is shape-only and takes
/// neither, so a selector without a runtime in hand can call it.
///
/// The heuristic itself is not matmul's: convolution and attention pick an
/// instruction the same way, so it lives beside the size types at the root of
/// `multi_level` and takes the device's capabilities as closures. Only the error
/// is ours.
#[allow(clippy::type_complexity)]
pub fn find_instruction_size<R, IsSupported, SupportedSizes>(
    client: &ComputeClient<R>,
    elems: (ElemType, ElemType, ElemType),
    problem_size: MatmulProblemSize,
    forced: (Option<u32>, Option<u32>, Option<u32>),
    is_supported: IsSupported,
    supported_sizes: SupportedSizes,
) -> Result<TileSize, MatmulAvailabilityError>
where
    R: Runtime,
    IsSupported: Fn(&ComputeClient<R>, MmaConfig) -> bool,
    SupportedSizes: Fn(&ComputeClient<R>, ElemType, ElemType, ElemType) -> Vec<TileSize>,
{
    let (lhs, rhs, acc) = elems;
    crate::multi_level::find_instruction_size(
        problem_size,
        forced,
        |m, n, k| {
            is_supported(
                client,
                MmaConfig {
                    a_type: lhs,
                    b_type: rhs,
                    cd_type: acc,
                    m,
                    n,
                    k,
                },
            )
        },
        || supported_sizes(client, lhs, rhs, acc),
    )
    .ok_or(MatmulAvailabilityError::TileSizeNotFound)
}

fn selection_tiny<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
    tile_matmul: TileMatmulKind,
) -> BatchMatmulBlueprint {
    // If the K axis is big, we can leverage that.
    let pk = u32::min(problem.k as u32 / tile_size.k(), 8);
    let pk = u32::max(pk, 1);

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(PartitionSize::new(1, 1, pk))
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();
    let cube_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountStrategy::FromProblem,
    };

    let hypercube = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::SwizzleRow(2))
        .cube_count_strategy(cube_count_strategy)
        .build();

    BatchMatmulBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(hypercube)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select_unswizzled(
        strategy: MultiRowStrategy,
        plane_count: usize,
        instruction: (usize, usize),
        problem: (usize, usize),
        vector_sizes: (usize, usize),
    ) -> (usize, usize, usize) {
        select_size(
            strategy,
            plane_count,
            instruction.0,
            instruction.1,
            problem.0,
            problem.1,
            vector_sizes.0,
            vector_sizes.1,
            usize::MAX,
            usize::MAX,
        )
    }

    /// Regression test: when fewer planes are available than the requested
    /// rows-per-plane (e.g. a large `problem_m` asks for 2 rows but only 1 plane
    /// fits), `select_size` must not return `stage_size_m == 0`. The zero used to
    /// propagate into a degenerate `TilingScheme` (`stage_size.m == 0`) and panic
    /// with "attempt to divide by zero" in `BatchMatmulBlueprint::cube_launch_info`.
    /// Reproduces the rf-detr crash on the `m=1024, n=4, k=256` matmul.
    #[test]
    fn select_size_never_yields_zero_stage_size_m() {
        let plane_count = 1;
        let instruction_m = 8;
        let problem_m = 1024;
        let problem_n = 4;

        for strategy in [
            MultiRowStrategy::Always(2),
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
        ] {
            let (rows_per_plane, stage_size_m, _partition_shape_n) = select_unswizzled(
                strategy,
                plane_count,
                (instruction_m, 8),
                (problem_m, problem_n),
                (4, 4),
            );

            assert!(
                stage_size_m >= 1,
                "stage_size_m must be >= 1 (got {stage_size_m}) for {strategy:?} with plane_count={plane_count}"
            );
            assert!(rows_per_plane >= 1);
            // With a single plane we can only fit a single row per plane.
            assert!(rows_per_plane <= plane_count);
        }
    }

    #[test]
    fn select_size_balances_stage_work_for_narrow_n() {
        let (rows_per_plane, stage_size_m, partition_shape_n) =
            select_unswizzled(MultiRowStrategy::Never, 16, (16, 8), (65536, 64), (8, 8));

        assert_eq!(rows_per_plane, 1);
        assert_eq!(stage_size_m, 4);
        assert_eq!(partition_shape_n, 8);
    }

    #[test]
    fn select_size_covers_preferred_instruction_shapes() {
        let cases = [
            // Tall, wide, and square instruction preferences.
            ((32, 8), (65536, 64), (1, 2, 8)),
            ((8, 32), (64, 65536), (1, 16, 16)),
            ((16, 16), (1024, 1024), (1, 16, 16)),
        ];

        for ((instruction_m, instruction_n), (problem_m, problem_n), expected) in cases {
            let size = select_unswizzled(
                MultiRowStrategy::Never,
                16,
                (instruction_m, instruction_n),
                (problem_m, problem_n),
                (8, 8),
            );

            assert_eq!(size, expected);
        }
    }

    #[test]
    fn select_size_preserves_stage_m_when_n_uses_all_planes() {
        let power_of_two = select_unswizzled(
            MultiRowStrategy::Always(2),
            16,
            (32, 8),
            (65536, 1024),
            (8, 8),
        );
        let non_power_of_two =
            select_unswizzled(MultiRowStrategy::Never, 10, (32, 8), (65536, 80), (8, 8));

        assert_eq!(power_of_two, (2, 8, 16));
        assert_eq!(non_power_of_two, (1, 10, 10));
    }

    #[test]
    fn select_size_rounds_partition_n_to_power_of_two() {
        let (_, _, partition_shape_n) =
            select_unswizzled(MultiRowStrategy::Never, 16, (16, 8), (65536, 33), (8, 8));

        assert_eq!(partition_shape_n, 8);
    }

    #[test]
    fn select_size_respects_vector_widths() {
        let narrow_lhs =
            select_unswizzled(MultiRowStrategy::Never, 16, (32, 8), (65536, 64), (1, 8));
        let wide_lhs = select_unswizzled(MultiRowStrategy::Never, 16, (32, 8), (65536, 64), (8, 1));

        assert_eq!(narrow_lhs, (1, 1, 8));
        assert_eq!(wide_lhs, (1, 8, 8));
    }

    #[test]
    fn select_size_balances_with_divisible_power_of_two_geometry() {
        let cases = [
            (MultiRowStrategy::Never, 16, 8, 32, 64, 64),
            (MultiRowStrategy::Always(3), 8, 16, 16, 1024, 64),
            (MultiRowStrategy::Never, 10, 32, 8, 1024, 64),
        ];

        for (strategy, plane_count, instruction_m, instruction_n, problem_m, problem_n) in cases {
            let (_, stage_size_m, partition_shape_n) = select_unswizzled(
                strategy,
                plane_count,
                (instruction_m, instruction_n),
                (problem_m, problem_n),
                (8, 8),
            );

            assert!(partition_shape_n.is_multiple_of(stage_size_m));
        }
    }

    #[test]
    fn select_size_applies_swizzle_limits_independently() {
        let max_stage_tiles_m = max_swizzle_tiles(16, 2);
        let max_partition_shape_n = max_swizzle_tiles(8, 4);
        let (rows_per_plane, stage_size_m, partition_shape_n) = select_size(
            MultiRowStrategy::Always(2),
            16,
            16,
            8,
            65536,
            128,
            8,
            1,
            max_stage_tiles_m,
            max_partition_shape_n,
        );

        assert_eq!(rows_per_plane, 2);
        assert_eq!(stage_size_m, 2);
        assert_eq!(partition_shape_n, 4);
        let max_swizzle_span = SwizzleMode::B128.span_size();
        assert!(rows_per_plane * stage_size_m * 16 * 2 <= max_swizzle_span);
        assert!(partition_shape_n * 8 * 4 <= max_swizzle_span);
    }
}
