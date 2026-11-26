//! Rows side by side across the reduced dim, written to a buffer that steps them along another axis:
//! no line of rows goes out whole, so the input is read a value at a time. The `reduce_dim` matrix
//! writes these rows as they lie.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::ReduceOperation;
use cubek_reduce::kernels::{Reduce, ReduceElements, ReduceOpStrategy};
use cubek_test_utils::{HostDataType, StridedLayout, TestInput};

use crate::reduce::it::test_case::{RULE, UNIT, is_the_references, ran};

/// A sum along the first dim of `[8, 2, 4]`, its cells column-major.
fn sums_into_column_major_cells(strategy: ReduceOpStrategy) {
    const VALUE_COUNT: usize = 8;
    const ROW_COUNT: usize = 2;
    const COLUMN_COUNT: usize = 4;
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let (input, input_host) = TestInput::builder(
        client.clone(),
        Shape::new([VALUE_COUNT, ROW_COUNT, COLUMN_COUNT]),
    )
    .dtype(dtype)
    .layout(StridedLayout::Explicit(vec![
        ROW_COUNT * COLUMN_COUNT,
        COLUMN_COUNT,
        1,
    ]))
    .arange()
    .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), Shape::new([1, ROW_COUNT, COLUMN_COUNT]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![
            ROW_COUNT * COLUMN_COUNT,
            1,
            ROW_COUNT,
        ]))
        .zeros()
        .generate_without_host_data();

    let launched = Reduce::new(strategy).execute(
        &client,
        input.binding(),
        out.clone().binding(),
        0,
        ReduceOperation::Sum,
        ReduceElements::for_operation(ReduceOperation::Sum, dtype, None),
    );
    if !ran(launched) {
        return;
    }

    is_the_references(
        &client,
        out,
        HostDataType::F32,
        &input_host,
        0,
        ReduceOperation::Sum,
    );
}

#[test]
fn the_unit_kernel_sums_into_column_major_cells() {
    sums_into_column_major_cells(UNIT);
}

#[test]
fn the_rule_sums_into_column_major_cells() {
    sums_into_column_major_cells(RULE);
}
