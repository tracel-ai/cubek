//! Forced-blueprint tests for non-default knobs (swizzle, partition
//! buffering). Each knob is exercised once on a representative routine —
//! per-routine combinations live in `full/`.

use cubek_convolution::ConvAlgorithm;
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_std::{PartitionSize, StageSize, SwizzleModes, stage::SwizzleMode};

use super::common::{default_size, default_tile_size, f16_dtypes, tiling_scheme};
use crate::suite::launcher_strategy::test_algo;

fn small_partition() -> PartitionSize {
    PartitionSize { m: 1, n: 1, k: 1 }
}

fn small_stage() -> StageSize {
    StageSize { m: 2, n: 2, k: 1 }
}

// -- Swizzle -----------------------------------------------------------------

fn run_swizzle(swizzle: SwizzleModes) {
    test_algo(
        ConvAlgorithm::SimpleSyncCyclic,
        f16_dtypes(),
        tiling_scheme(default_tile_size(), small_partition(), small_stage()),
        swizzle,
        PartitionBuffering::Single,
        default_size(),
    );
}

#[test]
fn swizzle_b32() {
    run_swizzle(SwizzleModes {
        lhs: SwizzleMode::B32,
        rhs: SwizzleMode::B32,
        ..Default::default()
    });
}

#[test]
fn swizzle_b64() {
    run_swizzle(SwizzleModes {
        lhs: SwizzleMode::B64,
        rhs: SwizzleMode::B64,
        ..Default::default()
    });
}

#[test]
fn swizzle_b128() {
    run_swizzle(SwizzleModes {
        lhs: SwizzleMode::B128,
        rhs: SwizzleMode::B128,
        ..Default::default()
    });
}

// -- Partition buffering -----------------------------------------------------
//
// Partition buffering pipelines inside a partition along n, so it needs at
// least two tiles along n (partition.n >= 2).

#[test]
fn partition_buffering_double() {
    test_algo(
        ConvAlgorithm::SimpleSyncTilewise,
        f16_dtypes(),
        tiling_scheme(
            default_tile_size(),
            PartitionSize { m: 1, n: 2, k: 1 },
            small_stage(),
        ),
        SwizzleModes::default(),
        PartitionBuffering::Double,
        default_size(),
    );
}
