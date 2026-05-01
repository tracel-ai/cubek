//! CPU reference and seeded "produce a HostData" primitives for matmul.
//!
//! - [`strategy_result`] runs the kernel once and returns its output as a
//!   [`HostData`].
//! - [`cpu_reference_result`] runs the naive triple-loop on the same seeded
//!   inputs and returns its output as a [`HostData`].

use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput, ValidationResult,
    assert_equals_approx, launch_and_capture_outcome,
};

use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    launch::{Strategy, launch_ref},
};

/// Run `strategy` against `problem` with seeded inputs and return its output as
/// a [`HostData`].
///
/// Inputs are generated via `TestInput::uniform` so the same `(problem, seeds)`
/// pair produces the same bits on every run.
pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    strategy: Strategy,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    produce_with(
        client,
        problem,
        seed_lhs,
        seed_rhs,
        |client, lhs, rhs, out, dtypes| launch_ref(&strategy, client, lhs, rhs, out, dtypes),
    )
}

/// CPU-only counterpart to [`strategy_result`]: generate the same seeded
/// inputs, run the naive triple-loop, return the result as a [`HostData`].
///
/// Slow on bench-scale problems by design — only useful as a ground truth.
pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let (_lhs, lhs_data, _rhs, rhs_data, _out, problem) =
        seed_inputs(&client, problem, seed_lhs, seed_rhs);
    Ok(matmul_cpu_reference(&lhs_data, &rhs_data, &problem))
}

fn produce_with<F>(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    seed_lhs: u64,
    seed_rhs: u64,
    launch: F,
) -> Result<HostData, String>
where
    F: FnOnce(
        &ComputeClient<TestRuntime>,
        InputBinding<TestRuntime>,
        InputBinding<TestRuntime>,
        cubecl::prelude::TensorBinding<TestRuntime>,
        &mut MatmulElems,
    ) -> Result<(), MatmulSetupError>,
{
    let (lhs, _lhs_data, rhs, _rhs_data, out, problem) =
        seed_inputs(&client, problem, seed_lhs, seed_rhs);

    let lhs_handle = InputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    let out_handle = out.clone().binding();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    let outcome = launch_and_capture_outcome(&client, |c| {
        launch(c, lhs_handle, rhs_handle, out_handle, &mut dtypes).into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            out,
            HostDataType::F32,
        )),
    }
}

type Tensor = TensorHandle<TestRuntime>;

fn seed_inputs(
    client: &ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    seed_lhs: u64,
    seed_rhs: u64,
) -> (Tensor, HostData, Tensor, HostData, Tensor, MatmulProblem) {
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

    (lhs, lhs_data, rhs, rhs_data, out, problem)
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

/// Same as [`assert_result`] but with an explicit epsilon.
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
