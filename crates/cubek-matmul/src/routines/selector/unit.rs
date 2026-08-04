use std::fmt::Display;

use crate::{
    components::{stage::PartitionBuffering, tile::TileMatmulKind},
    definition::{
        BatchMatmulBlueprint, MatmulElems, MatmulGlobalElems, MatmulKind, MatmulProblem,
        MatmulVectorSizes, SwizzleModes, TilingScheme,
    },
};
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{StorageType, VectorSize},
};
use cubek_std::{
    MatrixLayout,
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    stage::SwizzleMode,
};

#[derive(Default, Clone, Copy, Debug)]
pub enum TileSizeSelection {
    // Chooses the smallest tile size possible.
    MinTileSize,
    #[default]
    // Chooses the biggest tile size possible.
    MaxTileSize,
}

impl Display for TileSizeSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileSizeSelection::MinTileSize => f.write_str("min_tile_size"),
            TileSizeSelection::MaxTileSize => f.write_str("max_tile_size"),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub enum PartitionScaling {
    #[default]
    Enabled,
    Disabled,
}

#[derive(Default, Clone, Copy, Debug)]
pub enum StageScaling {
    Enabled(u8),
    #[default]
    Disabled,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct UnitTilingBlueprintOptions {
    pub tile: TileSizeSelection,
    pub stage: StageScaling,
    pub partition: PartitionScaling,
    pub swizzle: bool,
}

/// Computes a [BatchMatmulBlueprint] depending on the problem kind
pub fn infer_blueprint_unit<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    vector_sizes: &MatmulVectorSizes,
    options: UnitTilingBlueprintOptions,
    global_elems: &MatmulGlobalElems,
) -> (BatchMatmulBlueprint, MatmulElems) {
    let kind: MatmulKind = problem.into();
    let num_sms = client.properties().hardware.num_streaming_multiprocessors;
    // Per-cube shared-memory budget; the selectors cap the tiling so the chosen
    // blueprint never over-requests it (see `selection`).
    let max_smem = client.properties().hardware.max_shared_memory_size;
    let min_tile_size = usize::max(vector_sizes.lhs, vector_sizes.rhs);
    let min_tile_size = usize::max(vector_sizes.out, min_tile_size) as u32;
    let tile_size = u32::max(min_tile_size, 4);
    let dtypes = MatmulElems::from_globals(global_elems);

    let blueprint = match kind {
        MatmulKind::General => general_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::MatVec => matvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::VecMat => vecmat_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::ScalarVec => scalarvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::VecScalar => vecscalar_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::InnerProduct => inner_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::OuterProduct => outer_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::ScalarProduct => scalar_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_smem,
            options,
            &dtypes,
            vector_sizes,
        ),
    };

    (blueprint, dtypes)
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
#[allow(clippy::too_many_arguments)]
fn general_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    use cubek_std::MatrixLayout::*;

    // Manually tested for good performance on many shapes.
    let (tile_size, mut partition_size) =
        match (problem.lhs_layout, problem.rhs_layout, options.tile) {
            (RowMajor, _, TileSizeSelection::MinTileSize) => (
                (1, tile_size, tile_size),
                (
                    scale_partition(options.partition, problem.m, 4, 9),
                    2,
                    scale_partition(options.partition, problem.k, 2, 10),
                ),
            ),
            (ColMajor, RowMajor, TileSizeSelection::MinTileSize) => (
                (tile_size, tile_size, 1),
                (2, 2, scale_partition(options.partition, problem.k, 3, 10)),
            ),
            (ColMajor, ColMajor, _) | (_, _, TileSizeSelection::MaxTileSize) => (
                (tile_size, tile_size, tile_size),
                (
                    scale_partition(options.partition, problem.m, 2, 9),
                    2,
                    scale_partition(options.partition, problem.k, 2, 9),
                ),
            ),
        };

    let mut num_plane = 8;

    if double_buffering {
        if partition_size.0 > 2 {
            partition_size.0 /= 2;
        }
        if partition_size.2 > 2 {
            partition_size.2 /= 2;
        }
        num_plane /= 2;
    }

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane,
        },
        num_sms,
        max_smem,
        GlobalOrder::SwizzleRow(4),
        options.stage,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (M, K) @ (K, 1) → (M, 1)
#[allow(clippy::too_many_arguments)]
fn matvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (MatrixLayout::RowMajor, _) => ((1, 1, tile_size), (1, 1, tile_size * 2)),
        _ => ((tile_size, 1, tile_size), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: (plane_dim / 2).max(1),
            n: 2,
        },
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (1, K) @ (K, N) → (1, N)
#[allow(clippy::too_many_arguments)]
fn vecmat_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    let (tile_size, partition_size) = ((1, tile_size, tile_size), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: 2,
            n: (plane_dim / 2).max(1),
        },
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (1, 1) @ (1, N) → (1, N)
#[allow(clippy::too_many_arguments)]
fn scalarvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    use cubek_std::MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (RowMajor, ColMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (ColMajor, RowMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (ColMajor, ColMajor) => ((1, tile_size, tile_size), (2, 2, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: 2,
            n: (plane_dim / 2).max(1),
        },
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
#[allow(clippy::too_many_arguments)]
fn vecscalar_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    let (tile_size, partition_size) = ((tile_size, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: (plane_dim / 2).max(1),
            n: 2,
        },
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (1, K) @ (K, 1) → (1, 1)
#[allow(clippy::too_many_arguments)]
fn inner_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    use cubek_std::MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (RowMajor, ColMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (ColMajor, RowMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (ColMajor, ColMajor) => ((1, 1, tile_size), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: plane_dim, n: 1 }, // TODO: most planes does nothing.
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (M, 1) @ (1, N) → (M, N)
#[allow(clippy::too_many_arguments)]
fn outer_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    let (tile_size, partition_size) = ((tile_size, tile_size, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
#[allow(clippy::too_many_arguments)]
fn scalar_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    _tile_size: u32,
    num_sms: Option<u32>,
    max_smem: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> BatchMatmulBlueprint {
    let (tile_size, partition_size) = ((1, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane: 1,
        },
        num_sms,
        max_smem,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        problem,
        dtypes,
        vector_sizes,
        double_buffering,
    )
}

enum StageSelection {
    WithPlane { plane_dim: u32, num_plane: u32 },
    Fixed { m: u32, n: u32 },
}

impl StageSelection {
    fn into_stages(self) -> (u32, u32) {
        match self {
            StageSelection::WithPlane {
                plane_dim: plane_size,
                num_plane: num_planes,
            } => {
                let num_units = num_planes * plane_size;
                closest_factor_pair(num_units)
            }
            StageSelection::Fixed { m, n } => (m.max(1), n.max(1)), // non-zero
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn selection(
    t: (u32, u32, u32),
    p: (u32, u32, u32),
    buffering: PartitionBuffering,
    plane_dim: u32,
    stage: StageSelection,
    num_sms: Option<u32>,
    max_smem: usize,
    global_order: GlobalOrder,
    stage_scaling: StageScaling,
    swizzle: bool,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
    double_buffering: bool,
) -> BatchMatmulBlueprint {
    let (stage_size_m, stage_size_n) = stage.into_stages();

    debug_assert!(
        stage_size_m > 0 && stage_size_n > 0,
        "Invalid stage size after normalization: m={stage_size_m}, n={stage_size_n}"
    );

    let (stage_size_m, stage_size_n) = match stage_scaling {
        StageScaling::Enabled(f) => (stage_size_m / f as u32, stage_size_n / f as u32),
        StageScaling::Disabled => (stage_size_m, stage_size_n),
    };

    let stage_buffering = if double_buffering { 2 } else { 1 };

    // Shrink the register partition until the stage footprint fits the per-cube
    // shared-memory budget, else launch fails the `SharedMemoryTooBig` check on
    // shapes sharing an autotune bucket with the benchmarked one. Partition is
    // the safe knob: tile (vectorization) and stage (unit mapping) stay intact.
    let mut p = p;
    let tiling_scheme = loop {
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size(t.into())
            .with_partition_size(p.into())
            .with_stage_size((stage_size_m, stage_size_n, 1).into())
            .build()
            .unwrap();

        if unit_stage_smem_bytes(&tiling_scheme, dtypes, stage_buffering) <= max_smem
            || !halve_largest_partition(&mut p)
        {
            break tiling_scheme;
        }
    };

    let cube_count_strategy = match num_sms {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: false,
        },
        None => CubeCountStrategy::Flattened,
    };

    let hypercube = HypercubeBlueprint::builder()
        .global_order(global_order)
        .cube_count_strategy(cube_count_strategy)
        .build();

    let mut builder =
        BatchMatmulBlueprint::builder(TileMatmulKind::Register, tiling_scheme, plane_dim, problem)
            .partition_buffering(buffering)
            .stage_buffering(stage_buffering)
            .hypercube_blueprint(hypercube);

    if swizzle {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m() as usize,
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k() as usize,
        };

        builder = builder.shared_swizzle(SwizzleModes {
            lhs: select_swizzle(lhs_swizzle_dim, dtypes.lhs_stage, vector_sizes.lhs),
            rhs: select_swizzle(rhs_swizzle_dim, dtypes.rhs_stage, vector_sizes.rhs),
            ..Default::default()
        })
    }

    builder.build()
}

/// All modes currently use atom size 16
const SWIZZLE_ATOM: usize = 16;

fn select_swizzle(swizzle_dim: usize, elem: StorageType, vector_size: VectorSize) -> SwizzleMode {
    // Can't swizzle if vector size > swizzle atom
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
        _ => SwizzleMode::B128,
    }
}

/// Shared-memory bytes one cube allocates for a unit blueprint with this tiling.
///
/// Mirrors `requested_smem_bytes` in the batch setup: the lhs and rhs operand
/// stages (each `num_stages = stage_buffering`) plus the output stage at the
/// writer's allocation — each writer stages one tile per partition, so the out
/// term is one tile per stage partition, independent of the partition size.
/// Keeping this in step with the launch-time check is what lets [`selection`]
/// cap the tiling to a blueprint the check will accept, without capping harder
/// than the check requires.
fn unit_stage_smem_bytes(
    tiling_scheme: &TilingScheme,
    dtypes: &MatmulElems,
    stage_buffering: u32,
) -> usize {
    let em = tiling_scheme.elements_per_stage_along_m() as usize;
    let ek = tiling_scheme.elements_per_stage_along_k() as usize;
    let en = tiling_scheme.elements_per_stage_along_n() as usize;
    let buf = stage_buffering as usize;
    let out_writer = (tiling_scheme.partitions_per_stage_along_m()
        * tiling_scheme.partitions_per_stage_along_n()
        * tiling_scheme.tile_size.m
        * tiling_scheme.tile_size.n) as usize;

    em * ek * buf * dtypes.lhs_stage.size()
        + ek * en * buf * dtypes.rhs_stage.size()
        + out_writer * dtypes.acc_stage.size()
}

/// Halve the largest partition dimension greater than 1, returning whether a
/// reduction was made. Once every dimension is 1 there is nothing left to trim.
fn halve_largest_partition(p: &mut (u32, u32, u32)) -> bool {
    let max = p.0.max(p.1).max(p.2);
    if max <= 1 {
        return false;
    }
    if p.0 == max {
        p.0 /= 2;
    } else if p.1 == max {
        p.1 /= 2;
    } else {
        p.2 /= 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The out stage is charged at the writer's allocation (one tile per stage
    /// partition), like `requested_smem_bytes`; charging the full stage would
    /// shrink partitions the launch-time check accepts.
    #[test]
    fn out_stage_charged_at_writers_allocation() {
        let scheme = TilingScheme::builder()
            .with_tile_size((4, 4, 4).into())
            .with_partition_size((4, 4, 2).into())
            .with_stage_size((4, 4, 1).into())
            .build()
            .unwrap();
        let dtypes = MatmulElems::new_deprecated::<f32>();

        // lhs 64x8 + rhs 8x64 f32 stages, out = 4x4 partitions of one 4x4 tile.
        let expected = 64 * 8 * 4 + 8 * 64 * 4 + 16 * 16 * 4;
        assert_eq!(unit_stage_smem_bytes(&scheme, &dtypes, 1), expected);
    }
}

/// Returns the factor pair `(a, b)` of `n` minimizing their difference,
/// with `a >= b` and `a * b == n`.
pub fn closest_factor_pair(n: u32) -> (u32, u32) {
    let sqrt_n = (n as f64).sqrt() as u32;
    for a in (1..=sqrt_n).rev() {
        if n.is_multiple_of(a) {
            return (n / a, a);
        }
    }
    (n, 1)
}

fn scale_partition(setting: PartitionScaling, axis: usize, max_exp: u32, div_exp: u32) -> u32 {
    if let PartitionScaling::Disabled = setting {
        return 2u32.pow(max_exp);
    }

    let exp = u32::min((axis as u32 / 2u32.pow(div_exp)) + 1, max_exp);
    2u32.pow(exp)
}
