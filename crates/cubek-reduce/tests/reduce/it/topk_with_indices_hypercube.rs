//! The last-arrival hypercube on the fused reduce paths, values and indices from one launch.
//!
//! Its partials keep the values and their indices side by side, and its last cube merges them: a
//! row of 70001 values, an extent that is no power of two, splits into several parts at its first
//! setting, the last one short. The kernel skips where the device commits to no plane width or one
//! cube's stores do not reach another within a launch.

use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::ScopeChecker;

use crate::reduce::it::test_case::{HYPERCUBE, Op, ReduceScopeChecker, TestCase};

fn case() -> TestCase {
    TestCase::new::<f32>(
        Shape::new([3, 70001]),
        Strides::new(&[70001, 1]),
        Some(1),
        HYPERCUBE,
    )
}

#[test]
fn hypercube_max_with_indices() {
    ReduceScopeChecker.check([case().reduced_by(Op::MaxWithIndices)]);
}

#[test]
fn hypercube_min_with_indices() {
    ReduceScopeChecker.check([case().reduced_by(Op::MinWithIndices)]);
}

#[test]
fn hypercube_topk_with_indices_k3() {
    ReduceScopeChecker.check([case().reduced_by(Op::TopKWithIndices(3))]);
}

// The indices-only path, for comparison: a failure here means the breakage is not specific to
// writing both.
#[test]
fn hypercube_argtopk_k3() {
    ReduceScopeChecker.check([case().reduced_by(Op::ArgTopK(3))]);
}
