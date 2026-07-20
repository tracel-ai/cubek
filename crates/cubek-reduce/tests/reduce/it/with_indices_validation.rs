//! Argument validation for `reduce_with_indices`.
//!
//! The launch derives vectorization and write positions from the values tensor
//! alone, so an indices tensor with a different layout would be written *as if*
//! it had the values layout, producing silently wrong data. These tests pin
//! that such calls are rejected before launch instead.

use cubecl::{TestRuntime, config::autotune::AutotuneLevel, prelude::*, zspace::Shape};
use cubek_reduce::{
    ReduceError, ReduceStrategy, ReduceWithIndicesDtypes,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    reduce_with_indices,
    routines::{BlueprintStrategy, unit::UnitStrategy},
};
use cubek_test_utils::{StridedLayout, TestInput};

/// Run the fused top-k on a `[4, 8]` input with the given indices output
/// shape/strides and return the validation result. Values output is always the
/// valid contiguous `[4, 2]`.
fn try_launch(indices_shape: [usize; 2], indices_strides: Vec<usize>) -> Result<(), ReduceError> {
    let client = TestRuntime::client(&Default::default());
    let k = 2;

    let input_dtype = f32::as_type_native_unchecked().storage_type();
    let u32_dtype = u32::as_type_native_unchecked().storage_type();

    let input = TestInput::builder(client.clone(), Shape::new([4, 8]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![8, 1]))
        .zeros()
        .generate_without_host_data();
    let values = TestInput::builder(client.clone(), Shape::new([4, k]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![k, 1]))
        .zeros()
        .generate_without_host_data();
    let indices = TestInput::builder(client.clone(), Shape::new(indices_shape))
        .dtype(u32_dtype)
        .layout(StridedLayout::Explicit(indices_strides))
        .zeros()
        .generate_without_host_data();

    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
        autotune_level: AutotuneLevel::Full,
    };
    let dtypes = ReduceWithIndicesDtypes {
        input: input_dtype,
        values: input_dtype,
        indices: u32_dtype,
        accumulation: input_dtype,
    };

    reduce_with_indices::<TestRuntime>(
        &client,
        input.binding(),
        values.binding(),
        indices.binding(),
        1,
        strategy,
        ReduceOperationConfig::TopK(k),
        dtypes,
    )
}

#[test]
fn rejects_indices_with_mismatched_strides() {
    // Same shape and element count as the values output, but column-major.
    let result = try_launch([4, 2], vec![1, 4]);
    assert!(
        matches!(result, Err(ReduceError::MismatchIndicesStrides { .. })),
        "expected MismatchIndicesStrides, got {result:?}"
    );
}

#[test]
fn rejects_indices_with_mismatched_shape() {
    let result = try_launch([2, 4], vec![4, 1]);
    assert!(
        matches!(result, Err(ReduceError::MismatchIndicesShape { .. })),
        "expected MismatchIndicesShape, got {result:?}"
    );
}

#[test]
fn accepts_matching_outputs() {
    let result = try_launch([4, 2], vec![2, 1]);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}
