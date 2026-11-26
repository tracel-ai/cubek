use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::ScopeChecker;

use super::test_case::{
    CUBE, Op, PLANE, ReduceScopeChecker, TestCase, UNIT, ValuePlace, values_by_place,
};

/// `case` holding the values `data` lays out for its shape and axis.
fn with_values(case: TestCase, data: fn(&Shape, usize) -> Vec<f32>) -> TestCase {
    let values = data(&case.shape, case.axis.unwrap());
    case.with_data(values)
}

/// A test for each extremum of `case` over the values `data` lays out, each one launch.
macro_rules! extrema_tests {
    ($case:expr, $data:expr) => {
        use super::*;

        #[test]
        fn max() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::Max)]);
        }

        #[test]
        fn min() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::Min)]);
        }

        #[test]
        fn max_abs() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::MaxAbs)]);
        }

        #[test]
        fn arg_max() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::ArgMax)]);
        }

        #[test]
        fn arg_min() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::ArgMin)]);
        }

        #[test]
        fn max_with_indices() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::MaxWithIndices)]);
        }

        #[test]
        fn min_with_indices() {
            ReduceScopeChecker.check([with_values($case, $data).reduced_by(Op::MinWithIndices)]);
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
        unit_case(Shape::new([2, 64]), Strides::new(&[64, 1]), 1)
            .with_data(vec![f32::NEG_INFINITY; 128])
    }

    #[test]
    fn top_3_with_indices() {
        ReduceScopeChecker.check([case().reduced_by(Op::TopKWithIndices(3))]);
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
        nan_extrema_data
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

    fn case_holding_value(value: f32) -> TestCase {
        unit_case(Shape::new([2, 64]), Strides::new(&[64, 1]), 1).with_data(vec![value; 128])
    }

    #[test]
    fn max() {
        ReduceScopeChecker.check([case_holding_value(f32::NEG_INFINITY).reduced_by(Op::Max)]);
    }

    #[test]
    fn arg_max() {
        ReduceScopeChecker.check([case_holding_value(f32::NEG_INFINITY).reduced_by(Op::ArgMax)]);
    }

    #[test]
    fn max_with_indices() {
        ReduceScopeChecker
            .check([case_holding_value(f32::NEG_INFINITY).reduced_by(Op::MaxWithIndices)]);
    }

    #[test]
    fn min() {
        ReduceScopeChecker.check([case_holding_value(f32::INFINITY).reduced_by(Op::Min)]);
    }

    #[test]
    fn arg_min() {
        ReduceScopeChecker.check([case_holding_value(f32::INFINITY).reduced_by(Op::ArgMin)]);
    }

    #[test]
    fn min_with_indices() {
        ReduceScopeChecker
            .check([case_holding_value(f32::INFINITY).reduced_by(Op::MinWithIndices)]);
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
        ReduceScopeChecker.check([unit_case(Shape::new([2, 3]), Strides::new(&[3, 1]), 1)
            .with_data(VALUES.to_vec())
            .reduced_by(Op::MaxAbs)]);
    }

    #[test]
    fn plane() {
        ReduceScopeChecker.check([plane_case(Shape::new([2, 3]), Strides::new(&[3, 1]), 1)
            .with_data(VALUES.to_vec())
            .reduced_by(Op::MaxAbs)]);
    }
}

fn nan_extrema_data(shape: &Shape, axis: usize) -> Vec<f32> {
    values_by_place(shape, axis, |place| {
        let ValuePlace {
            coordinate,
            row,
            row_value_count,
        } = place;
        let last = row_value_count - 1;
        let base = (coordinate % 7) as f32 - 3.0 + (row % 3) as f32 * 0.125;

        match row % 9 {
            0 if coordinate == row_value_count / 2 => f32::NAN,
            1 if coordinate == 0 => f32::NAN,
            2 if coordinate == last => f32::NAN,
            3 if coordinate == row_value_count / 3 || coordinate == (2 * row_value_count) / 3 => {
                f32::NAN
            }
            4 => f32::NAN,
            5 if coordinate == 0 || coordinate == last => 9.0,
            6 if coordinate == 0 || coordinate == last => -9.0,
            7 if coordinate == 0 => f32::NEG_INFINITY,
            7 if coordinate == last => f32::INFINITY,
            7 if coordinate == row_value_count / 2 => f32::NAN,
            _ => base,
        }
    })
}

/// The NaN layout of [`nan_extrema_data`] with every NaN given its own payload and
/// sign, so that nothing but the coordinate can separate two of them.
fn mixed_nan_extrema_data(shape: &Shape, axis: usize) -> Vec<f32> {
    nan_extrema_data(shape, axis)
        .into_iter()
        .enumerate()
        .map(|(linear, value)| match value.is_nan() {
            true => nan_of_index(linear as u32),
            false => value,
        })
        .collect()
}

/// A NaN whose payload and sign are `index`'s own.
fn nan_of_index(index: u32) -> f32 {
    let payload = 0x7FC0_0000 | ((index % 0x3F_FFFF) + 1);
    let sign = (index % 2) << 31;
    f32::from_bits(payload | sign)
}

/// Both signed zeros in every row, above every other value, at coordinates that
/// interleave so each order of appearance is covered.
///
/// `-0.0` and `0.0` compare equal, so the coordinate breaks the tie. A ranking
/// that told the two apart would pass every other test here and pick the wrong
/// index for exactly this input.
fn signed_zero_data(shape: &Shape, axis: usize) -> Vec<f32> {
    values_by_place(shape, axis, |place| {
        match (place.coordinate + place.row) % 4 {
            0 => -0.0,
            1 => 0.0,
            _ => -((place.coordinate % 5) as f32) - 1.0,
        }
    })
}

fn integer_extrema_data(shape: &Shape, _axis: usize) -> Vec<f32> {
    (0..shape.iter().product())
        .map(|linear| ((linear * 17) % 31) as f32 - 15.0)
        .collect()
}
