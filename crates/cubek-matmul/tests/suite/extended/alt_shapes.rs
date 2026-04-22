//! Forced-blueprint tests covering non-aligned or skinny problem shapes.
//!
//! Uses `Strategy::SimpleCyclicCmma` as the representative routine; the goal
//! is to exercise the bounds-checking and vector-size handling of the loader
//! under awkward shapes, not to cover every routine.

use cubek_matmul::{launch::Strategy, routines::BlueprintStrategy};
use cubek_std::{PartitionSize, StageSize};

use super::common::{client, default_tile_size, f16_elems, plane_blueprint, problem, row_row};
use crate::suite::test_matmul_strategy;

fn run(m: usize, n: usize, k: usize) {
    let c = client();
    let p = problem(m, n, k, row_row(), f16_elems());
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
fn shape_100x100x100() {
    run(100, 100, 100);
}

#[test]
fn shape_100x99x100() {
    run(100, 99, 100);
}

#[test]
fn shape_100x100x99() {
    run(100, 100, 99);
}

#[test]
fn shape_1x256x256() {
    run(1, 256, 256);
}

#[test]
fn shape_23x1x17() {
    run(23, 1, 17);
}
