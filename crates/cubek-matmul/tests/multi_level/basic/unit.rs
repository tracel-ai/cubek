//! Inferred-blueprint smoke tests for unit-based routines.
//!
//! The interleaved routine is not covered here because its tile matmul
//! requires `tile.k % plane_dim == 0`, which the inferred selector doesn't
//! enforce: a forced-blueprint variant lives in `extended/tiling_scheme.rs`.

use cubek_matmul::multi_level::{
    Strategy as MultiLevel,
    routines::{TileSizeSelection, batch::simple_unit::SimpleUnitSelectionArgs},
};
use cubek_std::MatrixLayout;

use crate::harness::{client, f16_elems, rect_with_layouts, square, test_matmul_strategy};

#[cfg(feature = "heavy")]
#[test]
fn simple_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        MultiLevel::SimpleUnit(Default::default()).into(),
    );
}

#[cfg(feature = "heavy")]
#[test]
fn double_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        MultiLevel::DoubleUnit(Default::default()).into(),
    );
}

/// Both operands of a row/col layout prefer the inner product, so a one-line register tile is
/// the case that transposes the column-major rhs at load to run as an outer product.
#[cfg(feature = "heavy")]
#[test]
fn simple_unit_vecmat_row_col() {
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
        MultiLevel::SimpleUnit(Default::default()).into(),
    );
}

/// The min tile of a general row/col layout is one row high, the same transposed load on a matrix.
#[cfg(feature = "heavy")]
#[test]
fn simple_unit_min_tile_row_col() {
    test_matmul_strategy(
        client(),
        rect_with_layouts(
            64,
            64,
            64,
            MatrixLayout::RowMajor,
            MatrixLayout::ColMajor,
            f16_elems(),
        ),
        MultiLevel::SimpleUnit(cubek_matmul::routine::BlueprintStrategy::Inferred(
            SimpleUnitSelectionArgs {
                tile_size: TileSizeSelection::MinTileSize,
            },
        ))
        .into(),
    );
}
