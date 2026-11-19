use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::skip_unless_f16_arithmetic;

use super::test_case::{CUBE, PLANE, TestCase, UNIT};

/// `case` holding the values `data` lays out for its shape and axis.
fn with_values(case: TestCase, data: fn(&Shape, usize) -> Vec<f32>) -> TestCase {
    let values = data(&case.shape, case.axis.unwrap());
    case.with_data(values)
}

/// A test for each extremum of `case` over the values `data` lays out, each one launch; with
/// `unless`, a skip where that check of the device says so.
macro_rules! extrema_tests {
    ($case:expr, $data:expr $(, unless $skip:path)?) => {
        use super::*;

        #[test]
        fn max() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_max();
        }

        #[test]
        fn min() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_min();
        }

        #[test]
        fn max_abs() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_max_abs();
        }

        #[test]
        fn arg_max() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_argmax();
        }

        #[test]
        fn arg_min() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_argmin();
        }

        #[test]
        fn max_with_indices() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_max_with_indices();
        }

        #[test]
        fn min_with_indices() {
            $(if $skip(&cubecl::test_device().client()) { return; })?
            with_values($case, $data).test_min_with_indices();
        }
    };
}

fn unit_case(shape: Shape, strides: Strides, axis: usize) -> TestCase {
    TestCase::new::<f32>(shape, strides, Some(axis), UNIT)
}

fn plane_case(shape: Shape, strides: Strides, axis: usize) -> TestCase {
    TestCase::new::<f32>(shape, strides, Some(axis), PLANE)
}

fn cube_case(shape: Shape, strides: Strides, axis: usize) -> TestCase {
    TestCase::new::<f32>(shape, strides, Some(axis), CUBE)
}

mod unit_parallel_f32 {
    extrema_tests!(
        unit_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        nan_extrema_data
    );
}

mod plane_parallel_f32 {
    extrema_tests!(
        plane_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        nan_extrema_data
    );
}

mod cube_parallel_f32 {
    extrema_tests!(
        cube_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        nan_extrema_data
    );
}

mod cube_parallel_mixed_nan_f32 {
    extrema_tests!(
        cube_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        mixed_nan_extrema_data
    );
}

mod unit_parallel_mixed_nan_f32 {
    extrema_tests!(
        unit_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        mixed_nan_extrema_data
    );
}

mod plane_parallel_mixed_nan_f32 {
    extrema_tests!(
        plane_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        mixed_nan_extrema_data
    );
}

/// Every value equals the value a slot is seeded with, so each one reaches the weakest slot
/// without passing it. A ranked insert that asks only for strictly better leaves all three slots
/// holding the seed's `u32::MAX` coordinate instead of 0, 1 and 2.
mod unit_seed_valued_f32 {
    use super::*;

    fn case() -> TestCase {
        unit_case(Shape::new([2, 64]), Strides::new(&[64, 1]), 1).with_data(vec![f32::MIN; 128])
    }

    #[test]
    fn top_3_with_indices() {
        case().test_topk_with_indices(3);
    }

    #[test]
    fn arg_max() {
        case().test_argmax();
    }
}

mod unit_parallel_signed_zero_f32 {
    extrema_tests!(
        unit_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        signed_zero_data
    );
}

mod plane_parallel_signed_zero_f32 {
    extrema_tests!(
        plane_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1),
        signed_zero_data
    );
}

mod plane_perpendicular_f32 {
    extrema_tests!(
        plane_case(Shape::new([64, 9]), Strides::new(&[9, 1]), 0),
        nan_extrema_data
    );
}

mod plane_strided_f32 {
    extrema_tests!(
        plane_case(Shape::new([9, 64]), Strides::new(&[128, 2]), 1),
        nan_extrema_data
    );
}

mod plane_parallel_f16 {
    extrema_tests!(
        TestCase::new::<half::f16>(Shape::new([9, 64]), Strides::new(&[64, 1]), Some(1), PLANE),
        nan_extrema_data,
        unless skip_unless_f16_arithmetic
    );
}

mod integer_extrema_control_i32 {
    extrema_tests!(
        TestCase::new::<i32>(Shape::new([9, 64]), Strides::new(&[64, 1]), Some(1), PLANE),
        integer_extrema_data
    );
}

mod unit_singleton_f32 {
    extrema_tests!(
        unit_case(Shape::new([9, 1]), Strides::new(&[1, 1]), 1),
        nan_extrema_data
    );
}

/// Rows of `-inf` for a max and of `inf` for a min: the identity each starts from is a real value
/// here, and the first coordinate must win.
mod unit_infinite_identities_f32 {
    use super::*;

    fn case(value: f32) -> TestCase {
        unit_case(Shape::new([2, 64]), Strides::new(&[64, 1]), 1).with_data(vec![value; 128])
    }

    #[test]
    fn max() {
        case(f32::NEG_INFINITY).test_max();
    }

    #[test]
    fn arg_max() {
        case(f32::NEG_INFINITY).test_argmax();
    }

    #[test]
    fn max_with_indices() {
        case(f32::NEG_INFINITY).test_max_with_indices();
    }

    #[test]
    fn min() {
        case(f32::INFINITY).test_min();
    }

    #[test]
    fn arg_min() {
        case(f32::INFINITY).test_argmin();
    }

    #[test]
    fn min_with_indices() {
        case(f32::INFINITY).test_min_with_indices();
    }
}

#[cfg(feature = "heavy")]
mod cube_f32 {
    use super::super::test_case::CUBE;
    extrema_tests!(
        TestCase::new::<f32>(Shape::new([9, 256]), Strides::new(&[256, 1]), Some(1), CUBE),
        nan_extrema_data
    );
}

/// A row of three holding a NaN: its magnitude is a NaN too.
mod max_abs_short_axis_nan {
    use super::*;

    const VALUES: [f32; 6] = [1.0, f32::NAN, -3.0, 1.0, -4.0, 3.0];

    #[test]
    fn unit() {
        unit_case(Shape::new([2, 3]), Strides::new(&[3, 1]), 1)
            .with_data(VALUES.to_vec())
            .test_max_abs();
    }

    #[test]
    fn plane() {
        plane_case(Shape::new([2, 3]), Strides::new(&[3, 1]), 1)
            .with_data(VALUES.to_vec())
            .test_max_abs();
    }
}

fn nan_extrema_data(shape: &Shape, axis: usize) -> Vec<f32> {
    let axis_len = shape[axis];
    let inner_len = shape.iter().skip(axis + 1).product::<usize>();
    let num_elements = shape.iter().product::<usize>();

    (0..num_elements)
        .map(|linear| {
            let axis_coordinate = (linear / inner_len) % axis_len;
            let outer = linear / (inner_len * axis_len);
            let inner = linear % inner_len;
            let slice = outer * inner_len + inner;
            let base = (axis_coordinate % 7) as f32 - 3.0 + (slice % 3) as f32 * 0.125;

            match slice % 9 {
                0 if axis_coordinate == axis_len / 2 => f32::NAN,
                1 if axis_coordinate == 0 => f32::NAN,
                2 if axis_coordinate + 1 == axis_len => f32::NAN,
                3 if axis_coordinate == axis_len / 3 || axis_coordinate == (2 * axis_len) / 3 => {
                    f32::NAN
                }
                4 => f32::NAN,
                5 if axis_coordinate == 0 || axis_coordinate + 1 == axis_len => 9.0,
                6 if axis_coordinate == 0 || axis_coordinate + 1 == axis_len => -9.0,
                7 if axis_coordinate == 0 => f32::NEG_INFINITY,
                7 if axis_coordinate + 1 == axis_len => f32::INFINITY,
                7 if axis_coordinate == axis_len / 2 => f32::NAN,
                _ => base,
            }
        })
        .collect()
}

/// The NaN layout of [`nan_extrema_data`] with every NaN given its own payload and
/// sign, so that nothing but the coordinate can separate two of them.
fn mixed_nan_extrema_data(shape: &Shape, axis: usize) -> Vec<f32> {
    nan_extrema_data(shape, axis)
        .into_iter()
        .enumerate()
        .map(|(linear, value)| {
            if !value.is_nan() {
                return value;
            }
            let payload = 0x7FC0_0000 | ((linear as u32 % 0x3F_FFFF) + 1);
            let sign = if linear % 2 == 0 { 0 } else { 0x8000_0000 };
            f32::from_bits(payload | sign)
        })
        .collect()
}

/// Both signed zeros in every row, above every other value, at coordinates that
/// interleave so each order of appearance is covered.
///
/// `-0.0` and `0.0` compare equal, so the coordinate breaks the tie. A ranking
/// that told the two apart would pass every other test here and pick the wrong
/// index for exactly this input.
fn signed_zero_data(shape: &Shape, axis: usize) -> Vec<f32> {
    let axis_len = shape[axis];
    let inner_len = shape.iter().skip(axis + 1).product::<usize>();
    let num_elements = shape.iter().product::<usize>();

    (0..num_elements)
        .map(|linear| {
            let axis_coordinate = (linear / inner_len) % axis_len;
            let outer = linear / (inner_len * axis_len);
            let inner = linear % inner_len;
            let slice = outer * inner_len + inner;

            match (axis_coordinate + slice) % 4 {
                0 => -0.0,
                1 => 0.0,
                _ => -((axis_coordinate % 5) as f32) - 1.0,
            }
        })
        .collect()
}

fn integer_extrema_data(shape: &Shape, _axis: usize) -> Vec<f32> {
    (0..shape.iter().product())
        .map(|linear| ((linear * 17) % 31) as f32 - 15.0)
        .collect()
}
