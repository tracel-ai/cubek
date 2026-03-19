#![allow(missing_docs)]

#[cfg(feature = "extended")]
pub mod extended;
pub mod launcher;

mod reference;

pub(crate) use reference::assert_result;

use cubek_std::MatrixLayout;
use cubek_test_utils::StrideSpec;

pub(crate) fn layout_to_stride_spec(layout: MatrixLayout) -> StrideSpec {
    match layout {
        MatrixLayout::RowMajor => StrideSpec::RowMajor,
        MatrixLayout::ColMajor => StrideSpec::ColMajor,
    }
}
