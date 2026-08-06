use cubecl::{
    Runtime, TestRuntime,
    config::autotune::AutotuneLevel,
    features::{Plane, TypeUsage},
    frontend::CubePrimitive,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

use super::test_case::TestCase;

fn supports_plane() -> bool {
    let client = TestRuntime::client(&Default::default());
    client.properties().features.plane.contains(Plane::Ops)
}

fn supports_f16_arithmetic() -> bool {
    let client = TestRuntime::client(&Default::default());
    half::f16::supported_uses(&client).contains(TypeUsage::Arithmetic)
}

fn run_extrema(case: TestCase, data: Vec<f32>) {
    let case = case.with_data(data);
    case.test_max();
    case.test_min();
    case.test_max_abs();
    case.test_argmax();
    case.test_argmin();
    case.test_max_with_indices();
    case.test_min_with_indices();
}

fn run_nan_extrema(case: TestCase) {
    let data = nan_extrema_data(&case.shape, case.axis.unwrap());
    run_extrema(case, data);
}

fn strategy(routine: RoutineStrategy, parallel_output_vectorization: bool) -> ReduceStrategy {
    ReduceStrategy {
        autotune_level: AutotuneLevel::Full,
        vectorization: VectorizationStrategy {
            parallel_output_vectorization,
        },
        routine,
    }
}

fn unit_case(shape: Shape, strides: Strides, axis: usize) -> TestCase {
    TestCase::new::<f32>(
        shape,
        strides,
        Some(axis),
        strategy(
            RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            false,
        ),
    )
}

fn plane_case(
    shape: Shape,
    strides: Strides,
    axis: usize,
    independent: bool,
    vectorize_output: bool,
) -> TestCase {
    TestCase::new::<f32>(
        shape,
        strides,
        Some(axis),
        strategy(
            RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy { independent })),
            vectorize_output,
        ),
    )
}

#[test]
fn unit_parallel_f32() {
    run_nan_extrema(unit_case(Shape::new([9, 64]), Strides::new(&[64, 1]), 1));
}

#[test]
fn plane_parallel_f32() {
    if !supports_plane() {
        return;
    }
    run_nan_extrema(plane_case(
        Shape::new([9, 64]),
        Strides::new(&[64, 1]),
        1,
        false,
        true,
    ));
}

#[test]
fn plane_perpendicular_f32() {
    if !supports_plane() {
        return;
    }
    run_nan_extrema(plane_case(
        Shape::new([64, 9]),
        Strides::new(&[9, 1]),
        0,
        false,
        true,
    ));
}

#[test]
fn plane_strided_f32() {
    if !supports_plane() {
        return;
    }
    run_nan_extrema(plane_case(
        Shape::new([9, 64]),
        Strides::new(&[128, 2]),
        1,
        true,
        false,
    ));
}

#[test]
fn plane_parallel_f16() {
    if !supports_plane() || !supports_f16_arithmetic() {
        return;
    }
    run_nan_extrema(TestCase::new::<half::f16>(
        Shape::new([9, 64]),
        Strides::new(&[64, 1]),
        Some(1),
        strategy(
            RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                independent: true,
            })),
            true,
        ),
    ));
}

#[test]
fn integer_extrema_control_i32() {
    if !supports_plane() {
        return;
    }
    let shape = Shape::new([9, 64]);
    let data = integer_extrema_data(&shape);
    let case = TestCase::new::<i32>(
        shape,
        Strides::new(&[64, 1]),
        Some(1),
        strategy(
            RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                independent: true,
            })),
            true,
        ),
    );
    run_extrema(case, data);
}

#[test]
fn unit_singleton_f32() {
    run_nan_extrema(unit_case(Shape::new([9, 1]), Strides::new(&[1, 1]), 1));
}

#[test]
fn unit_infinite_identities_f32() {
    let case = |value| {
        unit_case(Shape::new([2, 64]), Strides::new(&[64, 1]), 1).with_data(vec![value; 128])
    };

    let max = case(f32::NEG_INFINITY);
    max.test_max();
    max.test_argmax();
    max.test_max_with_indices();

    let min = case(f32::INFINITY);
    min.test_min();
    min.test_argmin();
    min.test_min_with_indices();
}

#[cfg(feature = "heavy")]
mod heavy {
    use cubek_reduce::routines::cube::CubeStrategy;

    use super::*;

    fn case(use_planes: bool) -> TestCase {
        TestCase::new::<f32>(
            Shape::new([9, 256]),
            Strides::new(&[256, 1]),
            Some(1),
            strategy(
                RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy { use_planes })),
                true,
            ),
        )
    }

    #[test]
    fn cube_f32() {
        run_nan_extrema(case(false));
    }

    #[test]
    fn cube_with_planes_f32() {
        if !supports_plane() {
            return;
        }
        run_nan_extrema(case(true));
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

fn integer_extrema_data(shape: &Shape) -> Vec<f32> {
    (0..shape.iter().product())
        .map(|linear| ((linear * 17) % 31) as f32 - 15.0)
        .collect()
}
