//! A reduce whose output holds no element.
//!
//! A dimension of length zero outside the reduce axis leaves nothing to write, so the
//! launch is sized from zero working units. Where the plane count is read from the cpu
//! core count, a cube still holds at least one unit. These pin that such a call returns, through
//! the rule, which is what burn's unspecified reduce strategy asks for.

use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_reduce::{ReduceError, ReduceOperation};
use cubek_test_utils::{StridedLayout, TestInput};

use crate::reduce::it::test_case::RULE;

/// Every element of the reduction `f32`, its output's included.
fn elements() -> ReduceElements {
    let dtype = f32::elem_type_native();
    ReduceElements {
        input: dtype,
        accumulator: dtype,
        out: dtype,
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

    Reduce::new(RULE).execute(
        &client,
        &cubecl::test_device(),
        input.binding(),
        output.binding(),
        1,
        ReduceOperation::Sum,
        elements(),
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

    Reduce::new(RULE).execute_with_indices(
        &client,
        &cubecl::test_device(),
        input.binding(),
        values.binding(),
        indices.binding(),
        1,
        ReduceOperation::Max,
        elements(),
        index_dtype,
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
