//! Smoke tests for the simple sync-tilewise conv routine.

use cubek_convolution::ConvAlgorithm;
use cubek_matmul::definition::TilingScheme;
use cubek_std::PartitionSize;

use super::common::{default_partition_buffering, default_swizzle, f16_dtypes, small_size};
use crate::suite::{
    basic::common::{default_tile_size, small_stage},
    launcher_strategy::test_algo,
};

#[test]
fn simple_tilewise_cmma_small_f16() {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(default_tile_size())
        .with_partition_size(PartitionSize { m: 2, n: 2, k: 1 })
        .with_stage_size(small_stage())
        .build()
        .unwrap();

    test_algo(
        ConvAlgorithm::SimpleSyncTilewise,
        f16_dtypes(),
        tiling_scheme,
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
