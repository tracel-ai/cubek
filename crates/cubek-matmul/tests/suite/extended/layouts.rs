//! Forced-blueprint tests across the four lhs/rhs matrix-layout combinations.

use cubek_matmul::{launch::Strategy, routines::BlueprintStrategy};
use cubek_std::{MatrixLayout, PartitionSize, StageSize};

use super::common::{client, default_tile_size, f16_elems, plane_blueprint, problem};
use crate::suite::test_matmul_strategy;

fn run(layouts: (MatrixLayout, MatrixLayout)) {
    let c = client();
    let p = problem(256, 256, 256, layouts, f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        default_tile_size(),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
    test_matmul_strategy(
        c,
        p,
        Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn layout_row_row() {
    run((MatrixLayout::RowMajor, MatrixLayout::RowMajor));
}

#[test]
fn layout_row_col() {
    run((MatrixLayout::RowMajor, MatrixLayout::ColMajor));
}

#[test]
fn layout_col_row() {
    run((MatrixLayout::ColMajor, MatrixLayout::RowMajor));
}

#[test]
fn layout_col_col() {
    run((MatrixLayout::ColMajor, MatrixLayout::ColMajor));
}
