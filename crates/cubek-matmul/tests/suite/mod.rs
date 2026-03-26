#![allow(missing_docs)]

#[cfg(feature = "extended")]
pub mod extended;

mod launcher_routine;
mod launcher_strategy;
mod reference;

#[allow(deprecated)]
pub(crate) use launcher_routine::{InputRepresentation, test_matmul_routine};
pub(crate) use launcher_strategy::test_matmul_strategy;

pub(crate) use reference::assert_result;

use cubek_std::MatrixLayout;
use cubek_test_utils::StrideSpec;

pub(crate) fn layout_to_stride_spec(layout: MatrixLayout) -> StrideSpec {
    match layout {
        MatrixLayout::RowMajor => StrideSpec::RowMajor,
        MatrixLayout::ColMajor => StrideSpec::ColMajor,
    }
}
