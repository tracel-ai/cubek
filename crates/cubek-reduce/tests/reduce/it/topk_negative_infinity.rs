//! Top-k over rows holding fewer finite values than `k`.
//!
//! A real `-inf` must outrank every empty slot and every padding lane, so each
//! reported index stays inside the row. The rows hold from zero to `k + 1`
//! finite values, the rest `-inf`, and their lengths leave padding in vectors
//! and planes.

use cubecl::zspace::{Shape, Strides};
use cubek_reduce::kernels::ReduceOpStrategy;
use cubek_test_utils::skip_unless_f16_arithmetic;

use super::test_case::{CUBE, PLANE, TestCase, UNIT};

const K: usize = 4;

fn case(shape: Shape, strides: Strides, axis: usize, strategy: ReduceOpStrategy) -> TestCase {
    TestCase::new::<f32>(shape, strides, Some(axis), strategy)
}

/// `case` holding [`sparse_finite_data`] for a top-[`K`].
fn sparse(case: TestCase) -> TestCase {
    let data = sparse_finite_data(&case.shape, case.axis.unwrap(), K);
    case.with_data(data)
}

/// A test for each top-[`K`] of `case` over `data`, each one launch: its values, its indices, and
/// both at once; with `unless`, a skip where that check of the device says so.
macro_rules! top_k_tests {
    ($case:expr, $data:expr $(, unless $skip:path)?) => {
        use super::*;

        #[test]
        fn top_k() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            $data($case).test_topk(K);
        }

        #[test]
        fn arg_top_k() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            $data($case).test_argtopk(K);
        }

        #[test]
        fn top_k_with_indices() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            $data($case).test_topk_with_indices(K);
        }
    };
}

/// Slice `s` holds `s % (k + 2)` finite values at scattered coordinates and
/// `-inf` everywhere else, so slices with none, fewer than `k`, exactly `k` and
/// more than `k` finite values all occur.
fn sparse_finite_data(shape: &Shape, axis: usize, k: usize) -> Vec<f32> {
    let axis_len = shape[axis];
    let inner_len = shape.iter().skip(axis + 1).product::<usize>();
    let num_elements = shape.iter().product::<usize>();

    (0..num_elements)
        .map(|linear| {
            let axis_coordinate = (linear / inner_len) % axis_len;
            let outer = linear / (inner_len * axis_len);
            let inner = linear % inner_len;
            let slice = outer * inner_len + inner;
            let finite = slice % (k + 2);

            if (axis_coordinate * 7 + slice) % axis_len < finite {
                ((axis_coordinate * 5 + slice) % 11) as f32 * 0.25 - 1.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
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
        case(Shape::new([9, K]), Strides::new(&[K, 1]), 1, UNIT),
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
        case(Shape::new([9, K]), Strides::new(&[K, 1]), 1, PLANE),
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
        sparse,
        unless skip_unless_f16_arithmetic
    );
}

mod plane_parallel_f16 {
    top_k_tests!(
        TestCase::new::<half::f16>(Shape::new([9, 50]), Strides::new(&[50, 1]), Some(1), PLANE),
        sparse,
        unless skip_unless_f16_arithmetic
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
