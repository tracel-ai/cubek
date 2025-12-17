#![allow(missing_docs)]

pub mod layered;
pub mod naive;

mod reference;

use cubek_matmul::definition::MatrixLayout;
use cubek_test_utils::StrideSpec;
pub use reference::assert_result;

pub(crate) fn layout_to_stride_spec(layout: MatrixLayout) -> StrideSpec {
    match layout {
        MatrixLayout::RowMajor => StrideSpec::RowMajor,
        MatrixLayout::ColMajor => StrideSpec::ColMajor,
    }
}
