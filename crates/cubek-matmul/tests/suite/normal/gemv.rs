//! Inferred-blueprint smoke tests for the GEMV routines.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, rect};
use crate::suite::test_matmul_strategy;

#[test]
fn gemv_plane_parallel_vecmat() {
    test_matmul_strategy(
        client(),
        rect(1, 256, 256, f16_elems()),
        Strategy::GemvPlaneParallel(Default::default()),
    );
}

#[test]
fn gemv_unit_perpendicular_matvec() {
    test_matmul_strategy(
        client(),
        rect(256, 1, 256, f16_elems()),
        Strategy::GemvUnitPerpendicular(Default::default()),
    );
}
