//! CPU reference and high-level correctness helpers for matmul.
//!
//! Behind the `cpu-reference` feature so test code, benchmarks, and the tuner
//! can all share one source of truth. Every helper here is *non-panicking* —
//! callers (tests via `.enforce()`, the tuner via its own outcome types)
//! decide what to do with a `ValidationResult`.
//!
//! The CPU reference is a naive triple-loop, intended only for correctness
//! checking on small problems — never for production use.

use std::path::Path;

use cubecl::{
    TestRuntime, prelude::*, server::ServerError, std::tensor::TensorHandle, zspace::Shape,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput, TestOutcome,
    ValidationResult, assert_equals_approx, assert_equals_approx_from_file, write_host_data,
};

use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    launch::{Strategy, launch_ref},
};

/// What a `validate_*` helper should do once it has the kernel's output.
pub enum CorrectnessTarget<'a> {
    /// Compare against the in-process CPU reference.
    Cpu,
    /// Write the kernel output to `path` so a later run can validate against it.
    WriteReference(&'a Path),
    /// Read a previously-written reference at `path` and compare against it.
    ValidateReference(&'a Path),
}

/// One outcome from a high-level validation. `WroteReference` is its own
/// variant rather than a `Pass` — callers (the tuner) need to distinguish
/// "no comparison happened by design" from "passed".
#[derive(Debug)]
pub enum CorrectnessReport {
    /// Output matches the reference (CPU or file).
    Pass,
    /// Output does not match. String mirrors `ValidationResult::Fail`.
    Fail(String),
    /// Validation could not be performed (compile error, missing reference,
    /// dtype mismatch, etc).
    Error(String),
    /// Reference file was written. Only emitted in `WriteReference` mode.
    WroteReference,
}

/// Run a launchable strategy against `problem` with seeded inputs and produce
/// a [`CorrectnessReport`] according to `target`.
///
/// Inputs are generated via `TestInput::uniform` with `seed` so the same
/// `(problem, seed)` pair produces the same bits on every run — making it
/// safe to compare across commits via reference files.
pub fn validate_strategy(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    strategy: Strategy,
    seed_lhs: u64,
    seed_rhs: u64,
    epsilon: f32,
    target: CorrectnessTarget<'_>,
) -> CorrectnessReport {
    validate_with(
        client,
        problem,
        seed_lhs,
        seed_rhs,
        epsilon,
        target,
        move |client, lhs, rhs, out, dtypes| launch_ref(&strategy, client, lhs, rhs, out, dtypes),
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_with<F>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    seed_lhs: u64,
    seed_rhs: u64,
    epsilon: f32,
    target: CorrectnessTarget<'_>,
    launch: F,
) -> CorrectnessReport
where
    F: FnOnce(
        &ComputeClient<TestRuntime>,
        InputBinding<TestRuntime>,
        InputBinding<TestRuntime>,
        cubecl::prelude::TensorBinding<TestRuntime>,
        &mut MatmulElems,
    ) -> Result<(), MatmulSetupError>,
{
    let (lhs, lhs_data) = TestInput::builder(client.clone(), problem.lhs_shape.clone())
        .dtype(problem.global_dtypes.lhs)
        .stride(layout_to_stride_spec(problem.lhs_layout))
        .uniform(seed_lhs, -1., 1.)
        .generate_with_f32_host_data();
    let (rhs, rhs_data) = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(problem.global_dtypes.rhs)
        .stride(layout_to_stride_spec(problem.rhs_layout))
        .uniform(seed_rhs, -1., 1.)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(problem.global_dtypes.out)
        .stride(layout_to_stride_spec(MatrixLayout::RowMajor))
        .zeros()
        .generate_without_host_data();

    problem.lhs_strides = lhs.strides().clone();
    problem.rhs_strides = rhs.strides().clone();

    let lhs_handle = InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    let out_handle = out.clone().binding();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    let launch_outcome: ExecutionOutcome = get_server_error(&client)
        .unwrap_or(launch(&client, lhs_handle, rhs_handle, out_handle, &mut dtypes).into());
    let outcome = match launch_outcome {
        ExecutionOutcome::Executed => {
            get_server_error(&client).unwrap_or(ExecutionOutcome::Executed)
        }
        other => other,
    };

    match outcome {
        ExecutionOutcome::CompileError(e) => {
            CorrectnessReport::Error(format!("compile error: {e}"))
        }
        ExecutionOutcome::Executed => match target {
            CorrectnessTarget::Cpu => {
                let report = assert_result_with_epsilon(
                    &lhs_data, &rhs_data, &problem, &client, out, dtypes, epsilon,
                );
                from_validation(report)
            }
            CorrectnessTarget::WriteReference(path) => {
                let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                match write_host_data(path, &actual) {
                    Ok(_) => CorrectnessReport::WroteReference,
                    Err(e) => CorrectnessReport::Error(format!(
                        "write reference {}: {e}",
                        path.display()
                    )),
                }
            }
            CorrectnessTarget::ValidateReference(path) => {
                let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                from_validation(assert_equals_approx_from_file(&actual, path, epsilon))
            }
        },
    }
}

fn from_validation(v: ValidationResult) -> CorrectnessReport {
    match v {
        ValidationResult::Pass => CorrectnessReport::Pass,
        ValidationResult::Fail(reason) => CorrectnessReport::Fail(reason),
        ValidationResult::Error(reason) => CorrectnessReport::Error(reason),
        ValidationResult::Skipped(reason) => {
            CorrectnessReport::Error(format!("skipped: {reason}"))
        }
    }
}

/// Mirror of [`assert_equals_approx`] for tests that want a non-panicking
/// version. Same signature as the existing `assert_result` test helper.
pub fn assert_result(
    lhs: &HostData,
    rhs: &HostData,
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: TensorHandle<TestRuntime>,
    dtypes: MatmulElems,
) -> ValidationResult {
    let epsilon = matmul_epsilon(&dtypes, 500.);
    assert_result_with_epsilon(lhs, rhs, problem, client, out, dtypes, epsilon)
}

/// Same as [`assert_result`] but with an explicit epsilon. Used by the tuner
/// so the user can override the default tolerance.
pub fn assert_result_with_epsilon(
    lhs: &HostData,
    rhs: &HostData,
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: TensorHandle<TestRuntime>,
    _dtypes: MatmulElems,
    epsilon: f32,
) -> ValidationResult {
    let expected = matmul_cpu_reference(lhs, rhs, problem);
    let actual = HostData::from_tensor_handle(client, out, HostDataType::F32);
    assert_equals_approx(&actual, &expected, epsilon)
}

/// Default per-dtype epsilon × safety factor.
pub fn matmul_epsilon(elems: &MatmulElems, safety_factor: f32) -> f32 {
    let total_eps = elems
        .lhs_global
        .epsilon()
        .max(elems.rhs_global.epsilon())
        .max(elems.acc_global.epsilon())
        .max(elems.lhs_stage.epsilon())
        .max(elems.rhs_stage.epsilon())
        .max(elems.acc_stage.epsilon())
        .max(elems.lhs_register.epsilon())
        .max(elems.rhs_register.epsilon())
        .max(elems.acc_register.epsilon());

    total_eps as f32 * safety_factor
}

/// Naive CPU matmul. Slow on large payloads — intended only for testing.
pub fn matmul_cpu_reference(lhs: &HostData, rhs: &HostData, problem: &MatmulProblem) -> HostData {
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;

    let out_shape = problem.out_shape.clone();
    let rank = out_shape.len();
    let num_batches = problem.num_batches();

    let mut out = vec![0.0; num_batches * m * n];

    let mut batch_index = vec![0usize; rank - 2];
    let mut lhs_index = vec![0usize; rank];
    let mut rhs_index = vec![0usize; rank];
    let mut out_index = vec![0usize; rank];

    let lhs_batches = &problem.lhs_batches;
    let rhs_batches = &problem.rhs_batches;
    let out_batches = &problem.out_batches;

    for batch_flat in 0..num_batches {
        let mut t = batch_flat;
        for d in (0..rank - 2).rev() {
            batch_index[d] = t % out_batches[d];
            t /= out_batches[d];
        }

        for d in 0..rank - 2 {
            lhs_index[d] = if d < lhs_batches.len() && lhs_batches[d] != 1 {
                batch_index[d]
            } else {
                0
            };
            rhs_index[d] = if d < rhs_batches.len() && rhs_batches[d] != 1 {
                batch_index[d]
            } else {
                0
            };
            out_index[d] = batch_index[d];
        }

        for i in 0..m {
            lhs_index[rank - 2] = i;
            out_index[rank - 2] = i;

            for j in 0..n {
                rhs_index[rank - 1] = j;
                out_index[rank - 1] = j;

                let mut sum = 0.0;
                for kk in 0..k {
                    lhs_index[rank - 1] = kk;
                    rhs_index[rank - 2] = kk;

                    sum += lhs.get_f32(&lhs_index) * rhs.get_f32(&rhs_index);
                }

                let out_linear = batch_flat * (m * n) + i * n + j;
                out[out_linear] = sum;
            }
        }
    }

    let strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    HostData {
        data: HostDataVec::F32(out),
        shape: out_shape,
        strides,
    }
}

fn layout_to_stride_spec(layout: MatrixLayout) -> StrideSpec {
    match layout {
        MatrixLayout::RowMajor => StrideSpec::RowMajor,
        MatrixLayout::ColMajor => StrideSpec::ColMajor,
    }
}

/// Surface a server error as an `ExecutionOutcome` so the validation pipeline
/// can decide whether to skip the comparison instead of panicking. Mirrors
/// the helper used in tests/launcher_strategy.
fn get_server_error(client: &ComputeClient<TestRuntime>) -> Option<ExecutionOutcome> {
    use cubecl::server::{self, LaunchError};
    match client.flush() {
        Ok(_) => None,
        Err(ServerError::ServerUnhealthy { errors, .. }) => {
            for error in errors.iter() {
                if let server::ServerError::Launch(LaunchError::TooManyResources(_))
                | server::ServerError::Launch(LaunchError::CompilationError(_)) = error
                {
                    return Some(ExecutionOutcome::CompileError(format!("{errors:?}")));
                }
            }
            None
        }
        Err(err) => Some(ExecutionOutcome::CompileError(format!("{err:?}"))),
    }
}

// Silence "unused" warnings in the unlikely case TestOutcome / Shape become
// unused after API tweaks. Keeping them re-exported for now lets the runner
// reach types it might need without an extra `cubek-test-utils` dep.
#[allow(dead_code)]
fn _phantom_use(_: TestOutcome, _: Shape) {}
