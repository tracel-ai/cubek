//! The `Any` / `All` logical reductions over a mask whose rows hold none, every and one of their
//! values set, so each writes both a `0` and a `1`, which random inputs (`reduce_dim.rs`) seldom
//! do. The reference's answers on these rows are pinned by its own Small Tests.

use crate::reduce::it::test_case::UNIT;
use cubecl::{
    ir::{ElemType, FloatKind, UIntKind},
    prelude::*,
    zspace::Shape,
};
use cubek_reduce::ReduceOperation;
use cubek_reduce::eval::cpu_reference::reference;
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, StridedLayout, TestInput, TestOutcome,
    assert_equals_approx, launch_and_capture_outcome,
};

/// The mask's rows: none set, every one set, one set.
const ROWS: usize = 3;
/// Values a row of the mask holds.
const ROW_LEN: usize = 4;

/// `operation` over the mask's rows checked against the reference: the flags they write, or a
/// skip where the kernel does not compile.
fn reduces_the_mask_like_the_reference(operation: ReduceOperation) {
    let client = cubecl::test_device().client();

    let shape = Shape::new([ROWS, ROW_LEN]);
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 0.0, 0.0,
    ];

    let input_dtype = f32::elem_type_native();
    let (input_handle, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![ROW_LEN, 1]))
        .custom(data)
        .generate_with_f32_host_data();

    // Any/All write the flag storage (u32 here: the bool backing callers request on runtimes
    // without 8-bit storage, and the dtype every test runtime supports) while accumulating in the
    // input's element. The kernel writes the flags directly into the u32 output, so the
    // f32-input -> flag-output conversion is exercised end to end.
    let elements = ReduceElements {
        input: ElemType::Float(FloatKind::F32),
        accumulator: ElemType::Float(FloatKind::F32),
        out: ElemType::UInt(UIntKind::U32),
    };

    let output_handle = TestInput::builder(client.clone(), Shape::new([ROWS, 1]))
        .dtype(elements.out)
        .layout(StridedLayout::Explicit(vec![1, 1]))
        .zeros()
        .generate_without_host_data();

    let input_binding = input_handle.binding();
    let output_binding = output_handle.clone().binding();
    let outcome = launch_and_capture_outcome(&client, &[&output_handle.handle], |c| {
        Reduce::new(UNIT)
            .execute(
                c,
                &cubecl::test_device(),
                input_binding,
                output_binding,
                1,
                operation,
                elements,
            )
            .into()
    });

    match outcome {
        ExecutionOutcome::Executed => {
            let written = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
            let expected = reference(&input_host, 1, operation, None);
            assert_equals_approx(&written, &expected, 0.0)
                .as_test_outcome()
                .enforce();
        }
        ExecutionOutcome::CompileError(reason) => TestOutcome::CompileError(reason).enforce(),
    }
}

#[test]
fn any_writes_a_flag_a_row() {
    reduces_the_mask_like_the_reference(ReduceOperation::Any);
}

#[test]
fn all_writes_a_flag_a_row() {
    reduces_the_mask_like_the_reference(ReduceOperation::All);
}
