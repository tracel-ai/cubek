//! Smoke tests for the specialized async conv routines.

use cubek_convolution::ConvAlgorithm;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, small_size,
};
use crate::convolution::launcher_strategy::test_algo;

// `test_algo` pads the input, so im2col reads positions in the padding: the loads have to be
// bounds checked like any convolution read. Regression for the specialized routines copying
// with the matmul's unchecked async copy, an illegal memory access on padded inputs.

#[test]
fn specialized_async_cyclic_cmma_small_f16() {
    test_algo(
        ConvAlgorithm::SpecializedAsyncCyclic,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}

#[test]
fn specialized_async_strided_cmma_small_f16() {
    test_algo(
        ConvAlgorithm::SpecializedAsyncStrided,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
