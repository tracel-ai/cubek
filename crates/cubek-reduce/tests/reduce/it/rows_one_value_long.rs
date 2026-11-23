//! A reduction over an axis one value long, every row of the input one contiguous run: the
//! buffer takes wide lines, but a line along a row holds one value.
//!
//! The shape is pinned here rather than in the `reduce_dim` matrix, which also runs a top-k, whose
//! rows must hold at least `k` values.

use cubecl::zspace::{Shape, Strides};

use crate::reduce::it::test_case::{RULE, TestCase};

#[test]
pub fn a_mean_over_rows_one_value_long() {
    TestCase::new::<f32>(
        Shape::new([128, 1, 1]),
        Strides::new(&[1, 1, 1]),
        Some(2),
        RULE,
    )
    .test_mean();
}
