//! Top-k over rows holding fewer finite values than `k`.
//!
//! A real `-inf` must outrank every empty slot and every padding lane, so each
//! reported index stays inside the row. The rows hold from zero to `k + 1`
//! finite values, the rest `-inf`, and their lengths leave padding in vectors
//! and planes.

use cubecl::{
    config::autotune::AutotuneLevel,
    features::Plane,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

use super::test_case::TestCase;

const K: usize = 4;

fn supports_plane() -> bool {
    let client = cubecl::test_device().client();
    client.properties().features.plane.contains(Plane::Ops)
}

fn supports_f16_arithmetic() -> bool {
    cubek_test_utils::supports_f16_arithmetic(&cubecl::test_device().client())
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

fn unit() -> ReduceStrategy {
    strategy(
        RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        false,
    )
}

fn plane(independent: bool, vectorize_output: bool) -> ReduceStrategy {
    strategy(
        RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy { independent })),
        vectorize_output,
    )
}

fn cube(use_planes: bool) -> ReduceStrategy {
    strategy(
        RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy { use_planes })),
        false,
    )
}

fn case(shape: Shape, strides: Strides, axis: usize, strategy: ReduceStrategy) -> TestCase {
    TestCase::new::<f32>(shape, strides, Some(axis), strategy)
}

fn run_topk(case: TestCase, k: usize) {
    let data = sparse_finite_data(&case.shape, case.axis.unwrap(), k);
    let case = case.with_data(data);
    case.test_topk(k);
    case.test_argtopk(k);
    case.test_topk_with_indices(k);
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

#[test]
fn unit_parallel_f32() {
    run_topk(
        case(Shape::new([9, 12]), Strides::new(&[12, 1]), 1, unit()),
        K,
    );
}

#[test]
fn unit_perpendicular_f32() {
    run_topk(
        case(Shape::new([12, 9]), Strides::new(&[9, 1]), 0, unit()),
        K,
    );
}

#[test]
fn unit_k_equals_row_f32() {
    run_topk(
        case(Shape::new([9, K]), Strides::new(&[K, 1]), 1, unit()),
        K,
    );
}

#[test]
fn plane_parallel_f32() {
    if !supports_plane() {
        return;
    }
    for independent in [false, true] {
        run_topk(
            case(
                Shape::new([9, 50]),
                Strides::new(&[50, 1]),
                1,
                plane(independent, false),
            ),
            K,
        );
    }
}

#[test]
fn plane_perpendicular_f32() {
    if !supports_plane() {
        return;
    }
    run_topk(
        case(
            Shape::new([50, 9]),
            Strides::new(&[9, 1]),
            0,
            plane(false, true),
        ),
        K,
    );
}

#[test]
fn plane_strided_f32() {
    if !supports_plane() {
        return;
    }
    run_topk(
        case(
            Shape::new([9, 50]),
            Strides::new(&[128, 2]),
            1,
            plane(true, false),
        ),
        K,
    );
}

#[test]
fn plane_k_equals_row_f32() {
    if !supports_plane() {
        return;
    }
    run_topk(
        case(
            Shape::new([9, K]),
            Strides::new(&[K, 1]),
            1,
            plane(false, false),
        ),
        K,
    );
}

#[test]
fn cube_f32() {
    run_topk(
        case(Shape::new([9, 50]), Strides::new(&[50, 1]), 1, cube(false)),
        K,
    );
}

#[test]
fn cube_with_planes_f32() {
    if !supports_plane() {
        return;
    }
    run_topk(
        case(Shape::new([9, 50]), Strides::new(&[50, 1]), 1, cube(true)),
        K,
    );
}

#[test]
fn unit_parallel_f16() {
    if !supports_f16_arithmetic() {
        return;
    }
    run_topk(
        TestCase::new::<half::f16>(Shape::new([9, 12]), Strides::new(&[12, 1]), Some(1), unit()),
        K,
    );
}

#[test]
fn plane_parallel_f16() {
    if !supports_plane() || !supports_f16_arithmetic() {
        return;
    }
    run_topk(
        TestCase::new::<half::f16>(
            Shape::new([9, 50]),
            Strides::new(&[50, 1]),
            Some(1),
            plane(true, false),
        ),
        K,
    );
}

/// Integers have no `-inf`: their lowest value is both a real value and the
/// empty slot, and the lower coordinate still has to win.
#[test]
fn lowest_integer_rows_i32() {
    let shape = Shape::new([9, 50]);
    let strides = Strides::new(&[50, 1]);
    let mut strategies = vec![unit()];
    if supports_plane() {
        strategies.push(plane(false, false));
    }
    for strategy in strategies {
        let case = TestCase::new::<i32>(shape.clone(), strides.clone(), Some(1), strategy)
            .with_data(vec![i32::MIN as f32; 9 * 50]);
        case.test_topk(K);
        case.test_argtopk(K);
        case.test_topk_with_indices(K);
    }
}
