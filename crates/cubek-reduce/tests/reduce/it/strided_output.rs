//! Rows side by side across the reduced dim, written to a buffer that steps them along another axis:
//! no line of rows goes out whole, so the input is read a value at a time. The `reduce_dim` matrix
//! writes these rows as they lie.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::ReduceOperation;
use cubek_reduce::eval::cpu_reference::ReduceReference;
use cubek_reduce::kernels::{Reduce, ReduceElements, ReduceOpStrategy};
use cubek_test_utils::{HostData, HostDataType, StridedLayout, TestInput, assert_equals_approx};

use crate::reduce::it::test_case::{RULE, UNIT};

/// A sum along the first dim of `[8, 2, 4]`, its cells column-major.
fn sums_into_column_major_cells(strategy: ReduceOpStrategy) {
    const VALUES: usize = 8;
    const ROWS: usize = 2;
    const COLS: usize = 4;
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let (input, input_host) = TestInput::builder(client.clone(), Shape::new([VALUES, ROWS, COLS]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![ROWS * COLS, COLS, 1]))
        .arange()
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), Shape::new([1, ROWS, COLS]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![ROWS * COLS, 1, ROWS]))
        .zeros()
        .generate_without_host_data();

    Reduce::new(strategy)
        .execute(
            &client,
            &cubecl::test_device(),
            input.binding(),
            out.clone().binding(),
            0,
            ReduceOperation::Sum,
            ReduceElements::for_operation(ReduceOperation::Sum, dtype, None),
        )
        .expect("a sum writes its cells at their own strides");

    let written = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let expected = ReduceReference::new(&input_host, 0, ReduceOperation::Sum, None).output;
    assert_equals_approx(&written, &expected, 0.0)
        .as_test_outcome()
        .enforce();
}

#[test]
fn the_unit_kernel_sums_into_column_major_cells() {
    sums_into_column_major_cells(UNIT);
}

#[test]
fn the_rule_sums_into_column_major_cells() {
    sums_into_column_major_cells(RULE);
}
