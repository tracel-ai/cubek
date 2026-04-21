//! Inferred-blueprint smoke tests for the GEMV routines.

use cubek_matmul::launch::Strategy;
use cubek_std::MatrixLayout;

use super::common::{client, f16_elems, rect_with_layouts};
use crate::suite::test_matmul_strategy;

#[test]
fn gemv_plane_parallel_vecmat() {
    // GemvPlaneParallel on GPU requires ColMajor rhs for vec-mat problems.
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
        Strategy::GemvPlaneParallel(Default::default()),
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
        Strategy::GemvUnitPerpendicular(Default::default()),
    );
}
