//! Smoke tests for the specialized TMA conv routine.

use cubek_convolution::ConvAlgorithm;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, small_size,
};
use crate::suite::launcher_strategy::test_algo;

#[test]
fn specialized_tma_cmma_small_f16() {
    test_algo(
        ConvAlgorithm::SpecializedTma,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
