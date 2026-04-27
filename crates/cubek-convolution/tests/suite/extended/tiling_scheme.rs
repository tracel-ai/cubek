//! Per-routine `TilingScheme` sweep covering minimal/k-reduction/multi-plane
//! configurations. Uses `SimpleSyncCyclicConv` as the representative routine —
//! the full cartesian across other algorithm families lives in `full/`.

use cubek_convolution::kernels::algorithm::simple::SimpleSyncCyclicConv;
use cubek_matmul::components::tile_matmul::cmma::CmmaMatmul;
use cubek_matmul::{components::stage::PartitionBuffering, definition::SwizzleModes};
use cubek_std::{PartitionSize, StageSize};

use super::common::{default_size, default_tile_size, f16_dtypes, tiling_scheme};
use crate::suite::launcher_strategy::test_algo;

fn run(partition: PartitionSize, stage: StageSize) {
    test_algo::<SimpleSyncCyclicConv<CmmaMatmul>>(
        f16_dtypes(),
        tiling_scheme(default_tile_size(), partition, stage),
        SwizzleModes::default(),
        PartitionBuffering::Single,
        default_size(),
    );
}

#[test]
fn partition_1x1x1_stage_1x1x1() {
    run(
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
}

#[test]
fn partition_1x1x4_stage_1x1x1() {
    run(
        PartitionSize { m: 1, n: 1, k: 4 },
        StageSize { m: 1, n: 1, k: 1 },
    );
}

#[test]
fn partition_2x1x4_stage_2x2x1() {
    run(
        PartitionSize { m: 2, n: 1, k: 4 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

#[test]
fn partition_1x1x1_stage_4x4x1() {
    run(
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 4, n: 4, k: 1 },
    );
}
