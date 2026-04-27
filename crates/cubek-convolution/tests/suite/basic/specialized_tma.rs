//! Smoke tests for `SpecializedTmaConv`.

use cubek_convolution::kernels::algorithm::specialized::SpecializedTmaConv;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, small_size,
};
use crate::suite::launcher_strategy::test_algo;

#[test]
fn specialized_tma_cmma_small_f16() {
    test_algo::<SpecializedTmaConv>(
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
