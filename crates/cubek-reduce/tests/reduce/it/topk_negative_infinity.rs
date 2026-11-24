//! Top-k over rows holding fewer finite values than `k`.
//!
//! A real `-inf` must outrank every empty slot and every padding lane, so each
//! reported index stays inside the row. The rows hold from zero to `k + 1`
//! finite values, the rest `-inf`, and their lengths leave padding in vectors
//! and planes.

use cubecl::zspace::{Shape, Strides};
use cubek_reduce::kernels::ReduceOpStrategy;
use cubek_test_utils::ScopeChecker;

use super::test_case::{CUBE, Op, PLANE, ReduceScopeChecker, TestCase, UNIT, values_by_place};

const KEPT_COUNT: usize = 4;

fn case(shape: Shape, strides: Strides, axis: usize, strategy: ReduceOpStrategy) -> TestCase {
    TestCase::new::<f32>(shape, strides, Some(axis), strategy)
}

/// `case` holding [`sparse_finite_data`] for a top-[`KEPT_COUNT`].
fn sparse(case: TestCase) -> TestCase {
    let data = sparse_finite_data(&case.shape, case.axis.unwrap(), KEPT_COUNT);
    case.with_data(data)
}

/// A test for each top-[`KEPT_COUNT`] of `case` over `data`, each one launch: its values, its
/// indices, and both at once.
macro_rules! top_k_tests {
    ($case:expr, $data:expr) => {
        use super::*;

        #[test]
        fn top_k() {
            ReduceScopeChecker.check([$data($case).reduced_by(Op::TopK(KEPT_COUNT))]);
        }

        #[test]
        fn arg_top_k() {
            ReduceScopeChecker.check([$data($case).reduced_by(Op::ArgTopK(KEPT_COUNT))]);
        }

        #[test]
        fn top_k_with_indices() {
            ReduceScopeChecker.check([$data($case).reduced_by(Op::TopKWithIndices(KEPT_COUNT))]);
        }
    };
}

/// Row `s` holds `s % (k + 2)` finite values at scattered coordinates and
/// `-inf` everywhere else, so rows with none, fewer than `k`, exactly `k` and
/// more than `k` finite values all occur.
fn sparse_finite_data(shape: &Shape, axis: usize, k: usize) -> Vec<f32> {
    values_by_place(shape, axis, |place| {
        let finite = place.row % (k + 2);
        match (place.coordinate * 7 + place.row) % place.row_value_count < finite {
            true => ((place.coordinate * 5 + place.row) % 11) as f32 * 0.25 - 1.0,
            false => f32::NEG_INFINITY,
        }
    })
}

mod unit_parallel_f32 {
    top_k_tests!(
        case(Shape::new([9, 12]), Strides::new(&[12, 1]), 1, UNIT),
        sparse
    );
}

mod unit_perpendicular_f32 {
    top_k_tests!(
        case(Shape::new([12, 9]), Strides::new(&[9, 1]), 0, UNIT),
        sparse
    );
}

mod unit_k_equals_row_f32 {
    top_k_tests!(
        case(
            Shape::new([9, KEPT_COUNT]),
            Strides::new(&[KEPT_COUNT, 1]),
            1,
            UNIT
        ),
        sparse
    );
}

mod plane_parallel_f32 {
    top_k_tests!(
        case(Shape::new([9, 50]), Strides::new(&[50, 1]), 1, PLANE),
        sparse
    );
}

mod plane_perpendicular_f32 {
    top_k_tests!(
        case(Shape::new([50, 9]), Strides::new(&[9, 1]), 0, PLANE),
        sparse
    );
}

mod plane_strided_f32 {
    top_k_tests!(
        case(Shape::new([9, 50]), Strides::new(&[128, 2]), 1, PLANE),
        sparse
    );
}

mod plane_k_equals_row_f32 {
    top_k_tests!(
        case(
            Shape::new([9, KEPT_COUNT]),
            Strides::new(&[KEPT_COUNT, 1]),
            1,
            PLANE
        ),
        sparse
    );
}

mod cube_f32 {
    top_k_tests!(
        case(Shape::new([9, 50]), Strides::new(&[50, 1]), 1, CUBE),
        sparse
    );
}

mod unit_parallel_f16 {
    top_k_tests!(
        TestCase::new::<half::f16>(Shape::new([9, 12]), Strides::new(&[12, 1]), Some(1), UNIT),
        sparse
    );
}

mod plane_parallel_f16 {
    top_k_tests!(
        TestCase::new::<half::f16>(Shape::new([9, 50]), Strides::new(&[50, 1]), Some(1), PLANE),
        sparse
    );
}

/// `case` with every value the lowest `i32`.
fn lowest_integers(case: TestCase) -> TestCase {
    case.with_data(vec![i32::MIN as f32; 9 * 50])
}

/// Integers have no `-inf`: their lowest value is both a real value and the empty slot, and the
/// lower coordinate still has to win.
mod lowest_integer_rows_i32 {
    use super::*;

    mod unit {
        top_k_tests!(
            TestCase::new::<i32>(Shape::new([9, 50]), Strides::new(&[50, 1]), Some(1), UNIT),
            lowest_integers
        );
    }

    mod plane {
        top_k_tests!(
            TestCase::new::<i32>(Shape::new([9, 50]), Strides::new(&[50, 1]), Some(1), PLANE),
            lowest_integers
        );
    }
}
