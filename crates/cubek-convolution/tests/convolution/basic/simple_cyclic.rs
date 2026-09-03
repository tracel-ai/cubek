//! Smoke tests for the simple sync-cyclic conv routine.

use cubek_convolution::ConvAlgorithm;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, medium_size,
    small_size,
};
use crate::convolution::launcher_strategy::{test_algo, test_algo_asymmetric};

#[test]
fn simple_cyclic_cmma_small_f16() {
    test_algo(
        ConvAlgorithm::SimpleSyncCyclic,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}

#[test]
fn simple_cyclic_cmma_end_padding_f16() {
    test_algo_asymmetric(
        ConvAlgorithm::SimpleSyncCyclic,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}

#[cfg(feature = "heavy")]
#[test]
fn simple_cyclic_cmma_medium_f16() {
    test_algo(
        ConvAlgorithm::SimpleSyncCyclic,
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        medium_size(),
    );
}
