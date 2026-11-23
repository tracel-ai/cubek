//! `Reduce::execute_with_indices` writes its indices at their own strides, whatever the values'.
//! The calls outside its Domain are Small Tests in `kernels/operation.rs`, and the values and
//! indices it writes at matching strides are checked in `reduce_dim.rs`.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::ReduceOperation;
use cubek_reduce::eval::cpu_reference::reference;
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_test_utils::{HostData, HostDataType, StridedLayout, TestInput, assert_equals_approx};

use crate::reduce::it::test_case::UNIT;

/// A top-2 of `[4, 8]` rows counting up, its indices column-major beside row-major values: the
/// indices are the reference's, read at their own strides.
#[test]
fn writes_indices_at_their_own_strides() {
    const ROWS: usize = 4;
    const ROW_LEN: usize = 8;
    const K: usize = 2;
    let client = cubecl::test_device().client();
    let input_dtype = f32::elem_type_native();
    let u32_dtype = u32::elem_type_native();

    let (input, input_host) = TestInput::builder(client.clone(), Shape::new([ROWS, ROW_LEN]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![ROW_LEN, 1]))
        .arange()
        .generate_with_f32_host_data();
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
            values.binding(),
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

    let written = HostData::from_tensor_handle(&client, indices, HostDataType::from(u32_dtype));
    let expected = reference(&input_host, 1, ReduceOperation::ArgTopK { k: K }, None);
    assert_equals_approx(&written, &expected, 0.0)
        .as_test_outcome()
        .enforce();
}
