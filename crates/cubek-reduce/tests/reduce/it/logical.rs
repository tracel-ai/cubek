//! Deterministic coverage for the `Any` / `All` logical reductions.
//!
//! The randomized `test_any` / `test_all` cases (in `reduce_dim.rs`) compare the
//! GPU output against a CPU reference over many shapes/strategies, but a buggy
//! kernel could in principle agree with a buggy reference. These hand-written
//! cases pin the exact 0/1 outputs for a known mask, so both ops are guaranteed
//! to produce *both* a `0` and a `1` output.

use cubecl::{
    TestRuntime,
    ir::{ElemType, FloatKind, UIntKind},
    prelude::*,
    zspace::Shape,
};
use cubek_reduce::{
    ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    reduce,
    routines::{BlueprintStrategy, unit::UnitStrategy},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StridedLayout, TestInput,
    launch_and_capture_outcome,
};

/// Reduce a `[3, 4]` mask along `axis = 1` with `config` and return the 3 output
/// values. The rows are designed to hit every interesting slice:
/// `[0,0,0,0]` (empty), `[1,1,1,1]` (full), `[0,1,0,0]` (mixed).
fn reduce_mask(config: ReduceOperationConfig) -> Vec<f32> {
    let client = TestRuntime::client(&Default::default());

    let shape = Shape::new([3, 4]);
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0, // any = 0, all = 0
        1.0, 1.0, 1.0, 1.0, // any = 1, all = 1
        0.0, 1.0, 0.0, 0.0, // any = 1, all = 0
    ];

    let input_dtype = f32::as_type_native_unchecked().storage_type();
    let (input_handle, _) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .layout(StridedLayout::Explicit(vec![4, 1]))
        .custom(data)
        .generate_with_f32_host_data();

    // Drives the real `precision()` path: Any/All require the flag storage as
    // output (u32 here — the bool backing callers request on runtimes without
    // 8-bit storage, and the dtype every test runtime supports) while
    // accumulation stays = input. The kernel writes the flags directly into
    // the u32 output, so the f32-input -> flag-output conversion is exercised
    // end to end.
    let dtypes = config.precision(
        ElemType::Float(FloatKind::F32),
        Some(ElemType::UInt(UIntKind::U32)),
    );

    let output_handle = TestInput::builder(client.clone(), Shape::new([3, 1]))
        .dtype(dtypes.output)
        .layout(StridedLayout::Explicit(vec![1, 1]))
        .zeros()
        .generate_without_host_data();
    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };

    let input_binding = input_handle.binding();
    let output_binding = output_handle.clone().binding();
    let outcome = launch_and_capture_outcome(&client, |c| {
        reduce::<TestRuntime>(
            c,
            input_binding,
            output_binding,
            1,
            strategy,
            config,
            dtypes,
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
    let out = reduce_mask(ReduceOperationConfig::Any);
    assert_eq!(out, vec![0.0, 1.0, 1.0]);
}

#[test]
fn test_all_deterministic() {
    let out = reduce_mask(ReduceOperationConfig::All);
    assert_eq!(out, vec![0.0, 1.0, 0.0]);
}
