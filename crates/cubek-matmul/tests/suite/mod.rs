#![allow(missing_docs)]

pub mod basic;
#[cfg(feature = "extended")]
pub mod extended;
#[cfg(feature = "full")]
pub mod full;

mod bias;
pub(crate) mod launcher_strategy;
mod reference;

pub(crate) use launcher_strategy::test_matmul_strategy;

#[cfg(feature = "extended")]
pub(crate) use extended::test_matmul_test_strategy;

pub(crate) use reference::assert_result;

use cubek_std::MatrixLayout;
use cubek_test_utils::StrideSpec;

pub(crate) fn layout_to_stride_spec(layout: MatrixLayout) -> StrideSpec {
    match layout {
        MatrixLayout::RowMajor => StrideSpec::RowMajor,
        MatrixLayout::ColMajor => StrideSpec::ColMajor,
    }
}
