//! `Reduce::execute_with_indices` on calls inside its Domain: outputs that match,
//! min and max, and indices written at their own strides. The calls outside it
//! are Small Tests in `kernels/operation.rs`.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_reduce::{ReduceError, ReduceOperation};
use cubek_test_utils::{HostData, HostDataType, StridedLayout, TestInput};

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

/// A top-2 of `[4, 8]` rows counting up, its indices column-major beside row-major values: each
/// output is written at its own strides.
#[test]
fn writes_indices_at_their_own_strides() {
    const ROWS: usize = 4;
    const ROW_LEN: usize = 8;
    const K: usize = 2;
    let client = cubecl::test_device().client();
    let input_dtype = f32::elem_type_native();
    let u32_dtype = u32::elem_type_native();

    let input = TestInput::builder(client.clone(), Shape::new([ROWS, ROW_LEN]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![ROW_LEN, 1]))
        .arange()
        .generate_without_host_data();
    let values = TestInput::builder(client.clone(), Shape::new([ROWS, K]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![K, 1]))
        .zeros()
        .generate_without_host_data();
    let indices = TestInput::builder(client.clone(), Shape::new([ROWS, K]))
        .dtype(u32_dtype)
        .layout(StridedLayout::Explicit(vec![1, ROWS]))
        .zeros()
        .generate_without_host_data();

    Reduce::new(UNIT)
        .execute_with_indices(
            &client,
            &cubecl::test_device(),
            input.binding(),
            values.clone().binding(),
            indices.clone().binding(),
            1,
            ReduceOperation::TopK { k: K },
            ReduceElements {
                input: input_dtype,
                accumulator: input_dtype,
                out: input_dtype,
            },
            u32_dtype,
        )
        .expect("a top-k writes its indices at their own strides");

    let values = HostData::from_tensor_handle(&client, values, HostDataType::F32);
    let indices = HostData::from_tensor_handle(&client, indices, HostDataType::from(u32_dtype));
    for row in 0..ROWS {
        for slot in 0..K {
            // The largest values of a row counting up are its last ones, largest first.
            let index = ROW_LEN - 1 - slot;
            assert_eq!(
                indices.get_f32(&[row, slot]),
                index as f32,
                "row {row} slot {slot}"
            );
            assert_eq!(
                values.get_f32(&[row, slot]),
                (row * ROW_LEN + index) as f32,
                "row {row} slot {slot}"
            );
        }
    }
}

#[test]
fn accepts_matching_outputs() {
    let result = try_launch([4, 2], vec![2, 1]);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn accepts_min() {
    let result = try_launch_config(ReduceOperation::Min, 1, [4, 1], vec![1, 1]);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[test]
fn accepts_max() {
    let result = try_launch_config(ReduceOperation::Max, 1, [4, 1], vec![1, 1]);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}
