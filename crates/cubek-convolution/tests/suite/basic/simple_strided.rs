//! Smoke tests for `SimpleSyncStridedConv`.

use cubek_convolution::kernels::algorithm::simple::SimpleSyncStridedConv;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, medium_size,
    small_size,
};
use crate::suite::launcher_strategy::test_algo;

#[test]
fn simple_strided_cmma_small_f16() {
    test_algo::<SimpleSyncStridedConv>(
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}

#[cfg(feature = "basic")]
#[test]
fn simple_strided_cmma_medium_f16() {
    test_algo::<SimpleSyncStridedConv>(
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        medium_size(),
    );
}
