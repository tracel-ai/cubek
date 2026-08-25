//! Inferred-blueprint smoke tests for the GEMV routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;
use cubek_std::MatrixLayout;

use crate::harness::{client, f16_elems, rect_with_layouts, test_matmul_strategy};

#[cfg(feature = "heavy")]
#[test]
fn gemm_vecmat() {
    // Gemm handles VecMatCol (m=1, rhs ColMajor) on GPU via the Dot variant.
    test_matmul_strategy(
        client(),
        rect_with_layouts(
            1,
            128,
            128,
            MatrixLayout::RowMajor,
            MatrixLayout::ColMajor,
            f16_elems(),
        ),
        MultiLevel::Gemm(Default::default()).into(),
    );
}

#[test]
fn gemv_unit_perpendicular_vecmat() {
    // GemvUnitPerpendicular only accepts vec-mat shapes (m = 1).
    test_matmul_strategy(
        client(),
        rect_with_layouts(
            1,
            128,
            128,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            f16_elems(),
        ),
        MultiLevel::GemvUnitPerpendicular(Default::default()).into(),
    );
}
