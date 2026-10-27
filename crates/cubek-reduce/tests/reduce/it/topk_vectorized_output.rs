//! Top-k with a vectorized parallel output whose row count is not a multiple
//! of the output vector size.
//!
//! Each k slot of the output is one contiguous run of rows. With 9 rows and
//! `k = 4` the whole output holds 36 scalars, a multiple of 4, while a run of
//! 9 is not: a vector of 4 rows starting at row 8 would spill into the next
//! slot's run.

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

fn supports_plane() -> bool {
    let client = cubecl::test_device().client();
    client.properties().features.plane.contains(Plane::Ops)
}

fn vectorized(routine: RoutineStrategy) -> ReduceStrategy {
    ReduceStrategy {
        autotune_level: AutotuneLevel::Full,
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: true,
        },
        routine,
    }
}

fn run_topk(routine: RoutineStrategy, rows: usize, k: usize) {
    let case = TestCase::new::<f32>(
        Shape::new([rows, 64]),
        Strides::new(&[64, 1]),
        Some(1),
        vectorized(routine),
    );
    case.test_topk(k);
    case.test_argtopk(k);
    case.test_topk_with_indices(k);
}

fn routines() -> Vec<RoutineStrategy> {
    let mut routines = vec![
        RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: false,
        })),
    ];
    if supports_plane() {
        for independent in [false, true] {
            routines.push(RoutineStrategy::Plane(BlueprintStrategy::Inferred(
                PlaneStrategy { independent },
            )));
        }
    }
    routines
}

#[test]
fn rows_not_multiple_of_vector_k4() {
    for routine in routines() {
        run_topk(routine, 9, 4);
    }
}

#[test]
fn rows_not_multiple_of_vector_k2() {
    for routine in routines() {
        run_topk(routine, 6, 2);
    }
}

#[test]
fn rows_multiple_of_vector_k4() {
    for routine in routines() {
        run_topk(routine, 8, 4);
    }
}
