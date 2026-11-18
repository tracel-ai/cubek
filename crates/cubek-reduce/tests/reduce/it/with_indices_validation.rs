//! Argument validation for `Reduce::execute_with_indices`.
//!
//! The launch derives vectorization and write positions from the values tensor
//! alone, so an indices tensor with a different layout would be written *as if*
//! it had the values layout, producing silently wrong data. These tests pin
//! that such calls are rejected before launch instead.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_reduce::{ReduceError, ReduceOperation};
use cubek_test_utils::{StridedLayout, TestInput};

use crate::reduce::it::test_case::UNIT;

/// Run the fused top-k on a `[4, 8]` input with the given indices output
/// shape/strides and return the validation result. Values output is always the
/// valid contiguous `[4, 2]`.
fn try_launch(indices_shape: [usize; 2], indices_strides: Vec<usize>) -> Result<(), ReduceError> {
    try_launch_config(
        ReduceOperation::TopK { k: 2 },
        2,
        indices_shape,
        indices_strides,
    )
}

/// Run the fused `operation` on a `[4, 8]` input with the given indices output
/// shape/strides and return the validation result. Values output is always the
/// valid contiguous `[4, out_len]`.
fn try_launch_config(
    operation: ReduceOperation,
    out_len: usize,
    indices_shape: [usize; 2],
    indices_strides: Vec<usize>,
) -> Result<(), ReduceError> {
    let client = cubecl::test_device().client();

    let input_dtype = f32::elem_type_native();
    let u32_dtype = u32::elem_type_native();

    let input = TestInput::builder(client.clone(), Shape::new([4, 8]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![8, 1]))
        .zeros()
        .generate_without_host_data();
    let values = TestInput::builder(client.clone(), Shape::new([4, out_len]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![out_len, 1]))
        .zeros()
        .generate_without_host_data();
    let indices = TestInput::builder(client.clone(), Shape::new(indices_shape))
        .dtype(u32_dtype)
        .layout(StridedLayout::Explicit(indices_strides))
        .zeros()
        .generate_without_host_data();

    Reduce::new(UNIT).execute_with_indices(
        &client,
        &cubecl::test_device(),
        input.binding(),
        values.binding(),
        indices.binding(),
        1,
        operation,
        ReduceElements {
            input: input_dtype,
            accumulator: input_dtype,
            out: input_dtype,
        },
        u32_dtype,
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

#[test]
fn accepts_min_and_max() {
    for operation in [ReduceOperation::Min, ReduceOperation::Max] {
        let result = try_launch_config(operation, 1, [4, 1], vec![1, 1]);
        assert!(
            result.is_ok(),
            "expected Ok for {operation:?}, got {result:?}"
        );
    }
}

#[test]
fn rejects_operation_without_indices() {
    let result = try_launch_config(ReduceOperation::Sum, 1, [4, 1], vec![1, 1]);
    assert!(
        matches!(result, Err(ReduceError::IndicesUnsupported { .. })),
        "expected IndicesUnsupported, got {result:?}"
    );
}
