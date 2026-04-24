//! Smoke tests for `Strategy::Auto` across representative shapes and
//! precisions. Exercises the top-level selector end-to-end.

use cubek_matmul::launch::Strategy;

use super::common::{client, f16_elems, f32_elems, rect, square};
use crate::suite::test_matmul_strategy;

#[test]
fn auto_small_f16() {
    test_matmul_strategy(client(), square(16, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "basic")]
#[test]
fn auto_medium_f16() {
    test_matmul_strategy(client(), square(256, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "basic")]
#[test]
fn auto_medium_f32() {
    test_matmul_strategy(client(), square(256, f32_elems()), Strategy::Auto);
}

#[cfg(feature = "basic")]
#[test]
fn auto_skinny_vecmat() {
    test_matmul_strategy(client(), rect(1, 256, 256, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "basic")]
#[test]
fn auto_skinny_matvec() {
    test_matmul_strategy(client(), rect(256, 1, 256, f16_elems()), Strategy::Auto);
}
