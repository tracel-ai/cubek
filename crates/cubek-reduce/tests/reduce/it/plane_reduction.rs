//! An independent plane top-k over a single small vector.
//!
//! The shape, strides and strategy are pinned here rather than taken from the
//! `reduce_dim` matrix, which would run the identical reduction at every point
//! it instantiates.

use cubecl::{
    config::autotune::AutotuneLevel,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, plane::PlaneStrategy},
};

use crate::reduce::it::test_case::TestCase;

fn strategy() -> ReduceStrategy {
    ReduceStrategy {
        autotune_level: AutotuneLevel::Full,
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: true,
        },
        routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
            independent: true,
        })),
    }
}

#[test]
pub fn test_plane_reduction_simple_f32() {
    TestCase::new::<f32>(Shape::new([16]), Strides::new(&[1]), Some(0), strategy()).test_topk(3);
}

#[test]
pub fn test_plane_reduction_simple_f16() {
    TestCase::new::<half::f16>(Shape::new([16]), Strides::new(&[1]), Some(0), strategy())
        .test_topk(3);
}
