//! Inferred-blueprint smoke tests for unit-based routines.
//!
//! The interleaved routine is not covered here because its tile matmul
//! requires `tile.k % plane_dim == 0`, which the inferred selector doesn't
//! enforce: a forced-blueprint variant lives in `extended/tiling_scheme.rs`.

use cubek_matmul::multi_level::Strategy as MultiLevel;

use crate::harness::{client, f16_elems, square, test_matmul_strategy};

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
