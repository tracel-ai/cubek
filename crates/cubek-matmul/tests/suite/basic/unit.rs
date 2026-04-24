//! Inferred-blueprint smoke tests for unit-based routines.
//!
//! The interleaved routine is not covered here because its tile matmul
//! requires `tile.k % plane_dim == 0`, which the inferred selector doesn't
//! enforce — a forced-blueprint variant lives in `extended/tiling_scheme.rs`.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, square};
use crate::suite::test_matmul_strategy;

#[cfg(feature = "basic")]
#[test]
fn simple_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        Strategy::SimpleUnit(Default::default()),
    );
}

#[cfg(feature = "basic")]
#[test]
fn double_unit() {
    test_matmul_strategy(
        client(),
        square(64, f16_elems()),
        Strategy::DoubleUnit(Default::default()),
    );
}
