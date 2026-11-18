//! Top-k with a large `k`.
//!
//! A top-k insert walks all `k` slots for each candidate, and a merge walks them for each of the
//! other side's `k`. Unrolled unconditionally, the emitted kernel grows with `k²`: the `topk(300)`
//! an object-detection head asks for would expand to ~90k copies of the insertion body, which no
//! backend compiler returns from. These cases pin that a large `k` compiles and matches the host.
//!
//! `K` is only moderately large so the comparison stays cheap; the point is which code path runs,
//! not how large `k` gets.

use cubecl::zspace::{Shape, Strides};
use cubek_reduce::kernels::ReduceOpStrategy;

use crate::reduce::it::test_case::{PLANE, TestCase, UNIT};

/// Large enough that a `k`-by-`k` nest unrolled would be costly to compile.
const K: usize = 40;

fn case(strategy: ReduceOpStrategy) -> TestCase {
    TestCase::new::<f32>(
        Shape::new([4, 256]),
        Strides::new(&[256, 1]),
        Some(1),
        strategy,
    )
}

/// The case from the bug report: `k = 300` is what an object-detection head asks for. The failure
/// mode is a hang, not an assertion, so without a rolled loop this test does not finish rather than
/// reporting red.
#[test]
fn topk_300_terminates_and_matches_reference() {
    TestCase::new::<f32>(
        Shape::new([1, 1024]),
        Strides::new(&[1024, 1]),
        Some(1),
        UNIT,
    )
    .test_topk_with_indices(300);
}

#[test]
fn plane_topk_with_indices_large_k() {
    case(PLANE).test_topk_with_indices(K);
}

#[test]
fn plane_topk_large_k() {
    case(PLANE).test_topk(K);
}

#[test]
fn unit_topk_with_indices_large_k() {
    case(UNIT).test_topk_with_indices(K);
}

#[test]
fn unit_topk_large_k() {
    case(UNIT).test_topk(K);
}

#[test]
fn unit_argtopk_large_k() {
    case(UNIT).test_argtopk(K);
}
