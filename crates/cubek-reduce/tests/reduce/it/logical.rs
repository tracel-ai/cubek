//! Deterministic coverage for the `Any` / `All` logical reductions.
//!
//! The randomized `test_any` / `test_all` cases (in `reduce_dim.rs`) compare the
//! GPU output against a CPU reference over many shapes/strategies, but a buggy
//! kernel could in principle agree with a buggy reference. These hand-written
//! cases pin the exact 0/1 outputs for a known mask, so both ops are guaranteed
//! to produce *both* a `0` and a `1` output.

use crate::reduce::it::test_case::UNIT;
use cubecl::{
    ir::{ElemType, FloatKind, UIntKind},
    prelude::*,
    zspace::Shape,
};
use cubek_reduce::ReduceOperation;
use cubek_reduce::kernels::{Reduce, ReduceElements};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StridedLayout, TestInput,
    launch_and_capture_outcome,
};

/// Reduce a `[3, 4]` mask along `axis = 1` with `operation` and return the 3 output
/// values. The rows are designed to hit every interesting slice:
/// `[0,0,0,0]` (empty), `[1,1,1,1]` (full), `[0,1,0,0]` (mixed).
fn reduce_mask(operation: ReduceOperation) -> Vec<f32> {
    let client = cubecl::test_device().client();

    let shape = Shape::new([3, 4]);
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0, // any = 0, all = 0
        1.0, 1.0, 1.0, 1.0, // any = 1, all = 1
        0.0, 1.0, 0.0, 0.0, // any = 1, all = 0
    ];

    let input_dtype = f32::elem_type_native();
    let (input_handle, _) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![4, 1]))
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

    let output_handle = TestInput::builder(client.clone(), Shape::new([3, 1]))
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
            let host = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
            match host.data {
                HostDataVec::F32(values) => values,
                other => panic!("expected f32 output, got {other:?}"),
            }
        }
        ExecutionOutcome::CompileError(e) => panic!("compile error: {e}"),
    }
}

#[test]
fn test_any_deterministic() {
    let out = reduce_mask(ReduceOperation::Any);
    assert_eq!(out, vec![0.0, 1.0, 1.0]);
}

#[test]
fn test_all_deterministic() {
    let out = reduce_mask(ReduceOperation::All);
    assert_eq!(out, vec![0.0, 1.0, 0.0]);
}
