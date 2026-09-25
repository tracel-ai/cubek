//! A reduce whose output holds no element.
//!
//! A dimension of length zero outside the reduce axis leaves nothing to write.
//! The launch was still sized from it, which asks for zero working units, and
//! the routines divide the work by the units a cube holds. These pin that such
//! a call returns without launching.

use cubecl::{config::autotune::AutotuneLevel, prelude::*, zspace::Shape};
use cubek_reduce::{
    ReduceDtypes, ReduceError, ReduceStrategy, ReduceWithIndicesDtypes,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    reduce, reduce_with_indices,
    routines::{BlueprintStrategy, unit::UnitStrategy},
};
use cubek_test_utils::{StridedLayout, TestInput};

/// The unit routine is where the division lands, so it is the one forced here.
fn strategy() -> ReduceStrategy {
    ReduceStrategy {
        routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
        autotune_level: AutotuneLevel::Full,
    }
}

/// Sum the rows of a `[0, 8]` input into a `[0, 1]` output.
fn try_reduce() -> Result<(), ReduceError> {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let input = TestInput::builder(client.clone(), Shape::new([0, 8]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![8, 1]))
        .zeros()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), Shape::new([0, 1]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![1, 1]))
        .zeros()
        .generate_without_host_data();

    reduce(
        &client,
        input.binding(),
        output.binding(),
        1,
        strategy(),
        ReduceOperationConfig::Sum,
        ReduceDtypes {
            input: dtype,
            output: dtype,
            accumulation: dtype,
        },
    )
}

/// The same shapes through the fused path, which prepares its launch the same way.
fn try_reduce_with_indices() -> Result<(), ReduceError> {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let index_dtype = u32::elem_type_native();

    let input = TestInput::builder(client.clone(), Shape::new([0, 8]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![8, 1]))
        .zeros()
        .generate_without_host_data();
    let values = TestInput::builder(client.clone(), Shape::new([0, 1]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![1, 1]))
        .zeros()
        .generate_without_host_data();
    let indices = TestInput::builder(client.clone(), Shape::new([0, 1]))
        .dtype(index_dtype)
        .layout(StridedLayout::Explicit(vec![1, 1]))
        .zeros()
        .generate_without_host_data();

    reduce_with_indices(
        &client,
        input.binding(),
        values.binding(),
        indices.binding(),
        1,
        strategy(),
        ReduceOperationConfig::Max,
        ReduceWithIndicesDtypes {
            input: dtype,
            values: dtype,
            indices: index_dtype,
            accumulation: dtype,
        },
    )
}

#[test]
fn an_empty_output_is_not_launched() {
    let result = try_reduce();
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn an_empty_output_is_not_launched_with_indices() {
    let result = try_reduce_with_indices();
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}
