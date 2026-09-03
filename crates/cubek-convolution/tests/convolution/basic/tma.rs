//! Smoke tests for the simple async-TMA conv routine.

use cubek_convolution::ConvAlgorithm;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, small_size,
};
use crate::convolution::launcher_strategy::{test_algo, test_algo_asymmetric};

#[test]
fn simple_tma_cmma_small_f16() {
    test_algo(
        ConvAlgorithm::SimpleAsyncTma,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}

#[test]
fn simple_tma_cmma_end_padding_f16() {
    test_algo_asymmetric(
        ConvAlgorithm::SimpleAsyncTma,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
