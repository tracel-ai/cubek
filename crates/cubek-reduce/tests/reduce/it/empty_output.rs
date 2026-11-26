//! A reduce whose output holds no element.
//!
//! A dimension of length zero outside the reduce axis leaves nothing to write, so the
//! launch is sized from zero working units. Where the plane count is read from the cpu
//! core count, a cube still holds at least one unit. These pin that such a call returns, through
//! the rule, which is what burn's unspecified reduce strategy asks for.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::{prelude::*, zspace::Shape};
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_reduce::{ReduceError, ReduceOperation};
use cubek_test_utils::{StridedLayout, TestInput};

use crate::reduce::it::test_case::{RULE, ran};

/// Every element of the reduction `f32`, its output's included.
fn elements() -> ReduceElements {
    let dtype = f32::elem_type_native();
    ReduceElements {
        input: dtype,
        accumulator: dtype,
        out: dtype,
    }
}

/// A `[0, columns]` buffer of `dtype` on `client`.
fn empty(client: &Client, columns: usize, dtype: ElemType) -> TensorBinding {
    TestInput::builder(client.clone(), Shape::new([0, columns]))
        .dtype(dtype)
        .layout(StridedLayout::Explicit(vec![columns, 1]))
        .zeros()
        .generate_without_host_data()
        .binding()
}

/// Sum the rows of a `[0, 8]` input into a `[0, 1]` output.
fn try_reduce() -> Result<(), ReduceError> {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    Reduce::new(RULE).execute(
        &client,
        empty(&client, 8, dtype),
        empty(&client, 1, dtype),
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

    Reduce::new(RULE).execute_with_indices(
        &client,
        empty(&client, 8, dtype),
        empty(&client, 1, dtype),
        empty(&client, 1, index_dtype),
        1,
        ReduceOperation::Max,
        elements(),
        index_dtype,
    )
}

#[test]
fn a_call_whose_output_holds_nothing_returns() {
    ran(try_reduce());
}

#[test]
fn a_call_with_indices_whose_outputs_hold_nothing_returns() {
    ran(try_reduce_with_indices());
}
