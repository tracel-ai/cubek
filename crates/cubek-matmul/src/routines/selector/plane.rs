use cubecl::{
    {Runtime, client::ComputeClient, ir::StorageType},
    {features::MmaConfig, ir::VectorSize},
};
use cubek_std::{
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    stage::SwizzleMode,
    {MatmulProblemSize, MatrixLayout, PartitionSize, StageSize, TileSize},
};

use crate::definition::{
    MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes,
    MultiRowStrategy, SwizzleModes, TilingBlueprint, TilingScheme, adjust_dtypes,
};
use crate::routines::selector::is_tiny;
use crate::{
    components::global::{InputLoadFlow, LoadFlows},
    components::stage::PartitionBuffering,
    components::tile::TileMatmulFamily,
};

pub const NUM_SM_APPROX: u32 = 50;
pub const NUM_TENSOR_CORES_APPROX: u32 = 4;

#[derive(Default, Debug)]
/// Options to select the best plane matmul [selection](TilingBlueprint).
pub struct PlaneTilingBlueprintOptions {
    pub partition_k: Option<u32>,
    pub specialized: bool,
    pub swizzled: bool,
    pub row_count: Option<u32>,
    pub multi_row_strategy: MultiRowStrategy,
    pub partition_buffering: Option<PartitionBuffering>,
    /// Enables the tiny selector when the [matmul problem](MatmulProblem) is flagged as tiny.
    pub tiny_selection_enabled: bool,
}

pub fn infer_blueprint_plane<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    vector_sizes: &MatmulVectorSizes,
    options: PlaneTilingBlueprintOptions,
) -> Result<(TilingBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, TMM::requires_accelerator());

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
        TMM::is_supported,
        TMM::supported_sizes,
    )?;

    if options.tiny_selection_enabled && is_tiny(problem, &tile_size) {
        return Ok((
            selection_tiny(client, problem, tile_size, plane_dim),
            dtypes,
        ));
    }

    let row_count = options.row_count.unwrap_or_else(|| {
        let max_plane_per_cube = client.properties().hardware.max_units_per_cube / plane_dim;
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

    let (rows_per_plane, mut stage_size_m, mut partition_shape_n) = select_size(
        options.multi_row_strategy,
        row_count as usize,
        tile_size.m() as usize,
        problem.m,
    );

    if options.swizzled {
        if problem.lhs_layout == MatrixLayout::ColMajor {
            let elem_size = dtypes.lhs_global.size();
            while partition_shape_n * tile_size.n() as usize * elem_size > 128 {
                partition_shape_n /= 2;
                stage_size_m /= 2;
            }
        }
        if problem.rhs_layout == MatrixLayout::RowMajor {
            let elem_size = dtypes.rhs_global.size();
            while partition_shape_n * tile_size.n() as usize * elem_size > 128 {
                partition_shape_n /= 2;
                stage_size_m /= 2;
            }
        }
    }

    let mut partition_shape_k = options
        .partition_k
        .unwrap_or_else(|| plane_dim / tile_size.k());

    if options.swizzled {
        if problem.lhs_layout == MatrixLayout::RowMajor {
            let elem_size = dtypes.lhs_global.size() as u32;
            while partition_shape_k * tile_size.k() * elem_size > 128 {
                partition_shape_k /= 2;
            }
        }
        if problem.rhs_layout == MatrixLayout::ColMajor {
            let elem_size = dtypes.rhs_global.size() as u32;
            while partition_shape_k * tile_size.k() * elem_size > 128 {
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

    let mut builder = TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
        .partition_buffering(partition_buffering)
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

pub fn select_swizzle(
    swizzle_dim: usize,
    elem: StorageType,
    vector_size: VectorSize,
) -> SwizzleMode {
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

fn select_size(
    strategy: MultiRowStrategy,
    plane_count: usize,
    instruction_m: usize,
    problem_m: usize,
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

    (rows, plane_count / rows, plane_count)
}

/// A heuristic to choose the instruction to use, based on input shape
///
/// Will use 16x16 for balanced matrices, and 32x8 or 8x32 for degenerated ones.
#[allow(clippy::type_complexity)]
pub fn find_instruction_size<R, IsSupported, SupportedSizes>(
    client: &ComputeClient<R>,
    (lhs, rhs, acc): (StorageType, StorageType, StorageType),
    problem_size: MatmulProblemSize,
    (tm, tn, tk): (Option<u32>, Option<u32>, Option<u32>),
    is_supported: IsSupported,
    supported_sizes: SupportedSizes,
) -> Result<TileSize, MatmulAvailabilityError>
where
    R: Runtime,
    IsSupported: Fn(&ComputeClient<R>, MmaConfig) -> bool,
    SupportedSizes: Fn(&ComputeClient<R>, StorageType, StorageType, StorageType) -> Vec<TileSize>,
{
    let supported = |m: u32, n: u32, k: u32| {
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
    };

    let matches_forced = |m: u32, n: u32, k: u32| {
        tm.is_none_or(|v| m == v) && tn.is_none_or(|v| n == v) && tk.is_none_or(|v| k == v)
    };

    let is_valid = |m: u32, n: u32, k: u32| supported(m, n, k) && matches_forced(m, n, k);

    let try_candidate = |m: u32, n: u32, k: u32| {
        if is_valid(m, n, k) {
            Some(TileSize::from((m, n, k)))
        } else {
            None
        }
    };

    let (m, n) = (problem_size.m, problem_size.n);

    if m >= 4 * n
        && let Some(ts) = try_candidate(32, 8, 16)
    {
        return Ok(ts);
    }

    if n >= 4 * m
        && let Some(ts) = try_candidate(8, 32, 16)
    {
        return Ok(ts);
    }

    if let Some(ts) = try_candidate(16, 16, 16) {
        return Ok(ts);
    }

    if let Some(ts) = try_candidate(8, 8, 8) {
        return Ok(ts);
    }

    let val = supported_sizes(client, lhs, rhs, acc)
        .into_iter()
        .find(|ts| matches_forced(ts.m, ts.n, ts.k))
        .ok_or(MatmulAvailabilityError::TileSizeNotFound)?;

    Ok(val)
}

fn selection_tiny<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
) -> TilingBlueprint {
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

    TilingBlueprint::builder(tiling_scheme, plane_dim, problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(hypercube)
        .build()
}
