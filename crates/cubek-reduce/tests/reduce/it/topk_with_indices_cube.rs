//! The cube kernel on the fused reduce paths, values and indices from one launch.
//!
//! The `reduce_dim` matrix runs the cube kernel only under `heavy`, and the cube is the kernel
//! whose planes meet through shared memory, the values and their indices staged side by side. These
//! shapes are small so the cube kernels stay cheap to compile (the top-k insert and merge are
//! unrolled over `k`).

use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::ScopeChecker;

use crate::reduce::it::test_case::{CUBE, Op, ReduceScopeChecker, TestCase};

fn case() -> TestCase {
    TestCase::new::<f32>(Shape::new([4, 256]), Strides::new(&[256, 1]), Some(1), CUBE)
}

// `k > 1` is the interesting case: the staged carrier holds one slot per k, so a `k == 1` carrier
// stays correct even if the per-k slots are not merged.
#[test]
fn cube_topk_with_indices_k1() {
    ReduceScopeChecker.check([case().reduced_by(Op::TopKWithIndices(1))]);
}

#[test]
fn cube_topk_with_indices_k2() {
    ReduceScopeChecker.check([case().reduced_by(Op::TopKWithIndices(2))]);
}

#[test]
fn cube_topk_with_indices_k3() {
    ReduceScopeChecker.check([case().reduced_by(Op::TopKWithIndices(3))]);
}

#[test]
fn cube_topk_with_indices_k5() {
    ReduceScopeChecker.check([case().reduced_by(Op::TopKWithIndices(5))]);
}

#[test]
fn cube_min_with_indices() {
    ReduceScopeChecker.check([case().reduced_by(Op::MinWithIndices)]);
}

#[test]
fn cube_max_with_indices() {
    ReduceScopeChecker.check([case().reduced_by(Op::MaxWithIndices)]);
}

// The values-only and indices-only paths, for comparison: a failure here means the breakage is not
// specific to writing both.
#[test]
fn cube_topk_values_only_k3() {
    ReduceScopeChecker.check([case().reduced_by(Op::TopK(3))]);
}

#[test]
fn cube_argtopk_k3() {
    ReduceScopeChecker.check([case().reduced_by(Op::ArgTopK(3))]);
}
