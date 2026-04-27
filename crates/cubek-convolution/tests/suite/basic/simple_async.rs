//! Smoke tests for the async-copy simple algorithms.

use cubek_convolution::kernels::algorithm::simple::{
    SimpleAsyncCyclicConv, SimpleAsyncStridedConv,
};

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, small_size,
};
use crate::suite::launcher_strategy::test_algo;

#[test]
fn simple_async_cyclic_cmma_small_f16() {
    test_algo::<SimpleAsyncCyclicConv>(
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}

#[test]
fn simple_async_strided_cmma_small_f16() {
    test_algo::<SimpleAsyncStridedConv>(
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
