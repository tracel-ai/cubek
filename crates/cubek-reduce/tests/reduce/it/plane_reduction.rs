//! A plane top-k over a single small vector.
//!
//! The shape, strides and strategy are pinned here rather than taken from the `reduce_dim` matrix,
//! which would run the identical reduction at every point it instantiates.

use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::ScopeChecker;

use crate::reduce::it::test_case::{Op, PLANE, ReduceScopeChecker, TestCase};

#[test]
pub fn test_plane_reduction_simple_f32() {
    ReduceScopeChecker.check([TestCase::new::<f32>(
        Shape::new([16]),
        Strides::new(&[1]),
        Some(0),
        PLANE,
    )
    .reduced_by(Op::TopK(3))]);
}

#[test]
pub fn test_plane_reduction_simple_f16() {
    ReduceScopeChecker.check([TestCase::new::<half::f16>(
        Shape::new([16]),
        Strides::new(&[1]),
        Some(0),
        PLANE,
    )
    .reduced_by(Op::TopK(3))]);
}
