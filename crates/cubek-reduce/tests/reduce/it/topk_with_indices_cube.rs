//! The cube kernel on the fused reduce paths, values and indices from one launch.
//!
//! The `reduce_dim` matrix runs the cube kernel only under `heavy`, and the cube is the kernel
//! whose planes meet through shared memory, the values and their indices staged side by side. These
//! shapes are small so the cube kernels stay cheap to compile (the top-k insert and merge are
//! unrolled over `k`).

use cubecl::zspace::{Shape, Strides};

use crate::reduce::it::test_case::{CUBE, TestCase};

fn case() -> TestCase {
    TestCase::new::<f32>(Shape::new([4, 256]), Strides::new(&[256, 1]), Some(1), CUBE)
}

// `k > 1` is the interesting case: the staged carrier holds one slot per k, so a `k == 1` carrier
// stays correct even if the per-k slots are not merged.
#[test]
fn cube_topk_with_indices_k1() {
    case().test_topk_with_indices(1);
}

#[test]
fn cube_topk_with_indices_k2() {
    case().test_topk_with_indices(2);
}

#[test]
fn cube_topk_with_indices_k3() {
    case().test_topk_with_indices(3);
}

#[test]
fn cube_topk_with_indices_k5() {
    case().test_topk_with_indices(5);
}

#[test]
fn cube_min_with_indices() {
    case().test_min_with_indices();
}

#[test]
fn cube_max_with_indices() {
    case().test_max_with_indices();
}

// The values-only and indices-only paths, for comparison: a failure here means the breakage is not
// specific to writing both.
#[test]
fn cube_topk_values_only_k3() {
    case().test_topk(3);
}

#[test]
fn cube_argtopk_k3() {
    case().test_argtopk(3);
}
