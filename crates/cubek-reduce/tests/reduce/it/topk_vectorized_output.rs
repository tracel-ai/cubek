//! Top-k with a row count that is not a multiple of the output's vector width.
//!
//! Each k slot of the output is one contiguous run of rows. With 9 rows and `k = 4` the whole
//! output holds 36 scalars, a multiple of 4, while a run of 9 is not: a vector of 4 rows starting
//! at row 8 would spill into the next slot's run.

use cubecl::zspace::{Shape, Strides};
use cubek_reduce::kernels::ReduceOpStrategy;

use super::test_case::{CUBE, PLANE, RULE, TestCase, UNIT};

fn run_topk(strategy: ReduceOpStrategy, rows: usize, k: usize) {
    let case = TestCase::new::<f32>(
        Shape::new([rows, 64]),
        Strides::new(&[64, 1]),
        Some(1),
        strategy,
    );
    case.test_topk(k);
    case.test_argtopk(k);
    case.test_topk_with_indices(k);
}

/// The unit kernel and the rule, and the plane and cube kernels, which skip where the device has no
/// planes.
fn strategies() -> Vec<ReduceOpStrategy> {
    vec![UNIT, RULE, PLANE, CUBE]
}

#[test]
fn rows_not_multiple_of_vector_k4() {
    for strategy in strategies() {
        run_topk(strategy, 9, 4);
    }
}

#[test]
fn rows_not_multiple_of_vector_k2() {
    for strategy in strategies() {
        run_topk(strategy, 6, 2);
    }
}

#[test]
fn rows_multiple_of_vector_k4() {
    for strategy in strategies() {
        run_topk(strategy, 8, 4);
    }
}
