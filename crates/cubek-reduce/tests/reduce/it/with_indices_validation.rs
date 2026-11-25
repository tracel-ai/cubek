//! `Reduce::execute_with_indices` writes its indices at their own strides, whatever the values'.
//! The calls outside its Domain are Small Tests in `kernels/operation.rs`, and the values and
//! indices it writes at matching strides are checked in `reduce_dim.rs`.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::ReduceOperation;
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_test_utils::{HostDataType, StridedLayout, TestInput};

use crate::reduce::it::test_case::{UNIT, is_the_references, ran};

/// A top-2 of `[4, 8]` rows counting up, its indices column-major beside row-major values: the
/// indices are the reference's, read at their own strides.
#[test]
fn writes_indices_at_their_own_strides() {
    const ROW_COUNT: usize = 4;
    const ROW_VALUE_COUNT: usize = 8;
    const KEPT_COUNT: usize = 2;
    let client = cubecl::test_device().client();
    let input_dtype = f32::elem_type_native();
    let u32_dtype = u32::elem_type_native();

    let (input, input_host) =
        TestInput::builder(client.clone(), Shape::new([ROW_COUNT, ROW_VALUE_COUNT]))
            .dtype(input_dtype)
            .layout(StridedLayout::Explicit(vec![ROW_VALUE_COUNT, 1]))
            .arange()
            .generate_with_f32_host_data();
    let values = TestInput::builder(client.clone(), Shape::new([ROW_COUNT, KEPT_COUNT]))
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![KEPT_COUNT, 1]))
        .zeros()
        .generate_without_host_data();
    let indices = TestInput::builder(client.clone(), Shape::new([ROW_COUNT, KEPT_COUNT]))
        .dtype(u32_dtype)
        .layout(StridedLayout::Explicit(vec![1, ROW_COUNT]))
        .zeros()
        .generate_without_host_data();

    let launched = Reduce::new(UNIT).execute_with_indices(
        &client,
        input.binding(),
        values.binding(),
        indices.clone().binding(),
        1,
        ReduceOperation::TopK { k: KEPT_COUNT },
        ReduceElements {
            input: input_dtype,
            accumulator: input_dtype,
            out: input_dtype,
        },
        u32_dtype,
    );
    if !ran(launched) {
        return;
    }

    is_the_references(
        &client,
        indices,
        HostDataType::from(u32_dtype),
        &input_host,
        1,
        ReduceOperation::ArgTopK { k: KEPT_COUNT },
    );
}
