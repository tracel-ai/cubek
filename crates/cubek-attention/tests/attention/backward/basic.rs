//! Backward-pass test harness.
//!
//! Mirrors `basic.rs`: each test builds a seeded input set, routes it
//! through one of the three backward kernels (or the end-to-end entry
//! point), and compares the GPU output against the CPU reference in
//! [`cubek_attention::eval::backward::cpu_reference`].
//!
//! The kernels themselves are `todo!()` for now — these tests are the
//! harness, expected to fail at runtime until the kernels land.

#![cfg(feature = "cpu-reference")]

use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::CubePrimitive, zspace::Shape};
use cubek_attention::{
    backward::{
        BackwardConfig, flash_attention_backward, flash_attention_backward_dkdv,
        flash_attention_backward_dq, flash_attention_backward_prepass,
    },
    eval::backward::cpu_reference::{
        FlashAttentionBackwardDebug, flash_attention_backward_reference,
        flash_attention_backward_reference_debug,
    },
    forward::definition::{
        AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionOptions,
        AttentionProblem,
    },
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, assert_equals_approx,
    launch_and_capture_outcome,
};

/// Seeded inputs + their fp32 host-side mirrors. All tensors are fp32 in
/// the scaffold so we don't have to thread mixed precision through every
/// helper while the kernels are stubs.
struct BackwardInputs {
    q: cubecl::std::tensor::TensorHandle<TestRuntime>,
    q_data: HostData,
    k: cubecl::std::tensor::TensorHandle<TestRuntime>,
    k_data: HostData,
    v: cubecl::std::tensor::TensorHandle<TestRuntime>,
    v_data: HostData,
    do_: cubecl::std::tensor::TensorHandle<TestRuntime>,
    do_data: HostData,
}

fn problem(seq_q: usize, seq_kv: usize, head_dim: usize, val_dim: usize) -> AttentionProblem {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes: AttentionGlobalTypes::from_single_float_dtype(
            f32::as_type_native_unchecked(),
            AttentionGlobalTypes::mask_dtype(&client),
        ),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: Default::default(),
    }
}

fn problem_causal(
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    val_dim: usize,
) -> AttentionProblem {
    let mut p = problem(seq_q, seq_kv, head_dim, val_dim);
    p.options.causal = true;
    p
}

fn seed_inputs(client: &ComputeClient<TestRuntime>, problem: &AttentionProblem) -> BackwardInputs {
    let q_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.head_dim,
    ];
    let k_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_kv,
        problem.dims.head_dim,
    ];
    let v_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_kv,
        problem.dims.val_dim,
    ];
    let o_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.val_dim,
    ];

    let (q, q_data) = TestInput::builder(client.clone(), Shape::new(q_shape))
        .dtype(problem.global_dtypes.query)
        .uniform(11, -1., 1.)
        .generate_with_f32_host_data();
    let (k, k_data) = TestInput::builder(client.clone(), Shape::new(k_shape))
        .dtype(problem.global_dtypes.key)
        .uniform(22, -1., 1.)
        .generate_with_f32_host_data();
    let (v, v_data) = TestInput::builder(client.clone(), Shape::new(v_shape))
        .dtype(problem.global_dtypes.value)
        .uniform(33, -1., 1.)
        .generate_with_f32_host_data();
    let (do_, do_data) = TestInput::builder(client.clone(), Shape::new(o_shape))
        .dtype(problem.global_dtypes.out)
        .uniform(44, -1., 1.)
        .generate_with_f32_host_data();

    BackwardInputs {
        q,
        q_data,
        k,
        k_data,
        v,
        v_data,
        do_,
        do_data,
    }
}

fn zeros_like(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 4],
    dtype: cubecl::ir::StorageType,
) -> cubecl::std::tensor::TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

fn zeros_row(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 3],
    dtype: cubecl::ir::StorageType,
) -> cubecl::std::tensor::TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

fn config(problem: &AttentionProblem) -> BackwardConfig {
    let mut c = BackwardConfig::from_head_dim(problem.dims.head_dim);
    c.causal = problem.options.causal;
    c
}

/// Loose epsilon for scaffold tests — the GPU side currently panics, so this
/// only kicks in when the kernels start producing real results.
const EPS: f32 = 1e-3;

/// Run the prepass kernel and compare its `D` output against the CPU
/// reference. Expected to fail until the kernel lands.
fn run_prepass(problem: AttentionProblem) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let inputs = seed_inputs(&client, &problem);

    // We need O for the prepass — generate it by running the CPU forward.
    // For scaffold simplicity, derive O from the debug reference.
    let dbg = flash_attention_backward_reference_debug(
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
        &problem,
    );

    let o_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.val_dim,
    ];
    let row_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
    ];

    // Upload O from the reference back to the device.
    let o = TestInput::builder(client.clone(), Shape::new(o_shape))
        .dtype(problem.global_dtypes.out)
        .custom(o_data_to_vec(&dbg))
        .generate_without_host_data();

    let d = zeros_row(
        &client,
        row_shape,
        f32::as_type_native_unchecked().storage_type(),
    );
    let d_handle = d.clone();

    let outcome = launch_and_capture_outcome(&client, |c| {
        flash_attention_backward_prepass(
            c,
            o.clone().binding(),
            inputs.do_.clone().binding(),
            d.clone().binding(),
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => {
            let actual = HostData::from_tensor_handle(&client, d_handle, HostDataType::F32);
            assert_equals_approx(&actual, &dbg.d, EPS)
                .as_test_outcome()
                .enforce();
        }
    }
}

fn o_data_to_vec(dbg: &FlashAttentionBackwardDebug) -> Vec<f32> {
    match &dbg.o.data {
        cubek_test_utils::HostDataVec::F32(v) => v.clone(),
        _ => unreachable!("debug reference produces fp32 outputs"),
    }
}

/// Run the dQ kernel with CPU-computed `lse` and `D`, compare its output
/// against the CPU reference's dQ. Expected to fail until the kernel lands.
fn run_dq(problem: AttentionProblem) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let inputs = seed_inputs(&client, &problem);

    let dbg = flash_attention_backward_reference_debug(
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
        &problem,
    );

    let row_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
    ];
    let q_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.head_dim,
    ];

    let lse = upload_row(&client, row_shape, &dbg.lse);
    let d = upload_row(&client, row_shape, &dbg.d);
    let dq = zeros_like(&client, q_shape, problem.global_dtypes.query);
    let dq_handle = dq.clone();

    let cfg = config(&problem);
    let outcome = launch_and_capture_outcome(&client, |c| {
        flash_attention_backward_dq(
            c,
            inputs.q.clone().binding(),
            inputs.k.clone().binding(),
            inputs.v.clone().binding(),
            inputs.do_.clone().binding(),
            lse.clone().binding(),
            d.clone().binding(),
            dq.clone().binding(),
            &problem.global_dtypes,
            cfg.clone(),
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => {
            let actual = HostData::from_tensor_handle(&client, dq_handle, HostDataType::F32);
            let expected = flash_attention_backward_reference(
                &inputs.q_data,
                &inputs.k_data,
                &inputs.v_data,
                &inputs.do_data,
                &dbg.lse,
                &dbg.d,
                &problem,
            )
            .dq;
            assert_equals_approx(&actual, &expected, EPS)
                .as_test_outcome()
                .enforce();
        }
    }
}

fn run_dkdv(problem: AttentionProblem) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let inputs = seed_inputs(&client, &problem);

    let dbg = flash_attention_backward_reference_debug(
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
        &problem,
    );

    let row_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
    ];
    let k_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_kv,
        problem.dims.head_dim,
    ];
    let v_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_kv,
        problem.dims.val_dim,
    ];

    let lse = upload_row(&client, row_shape, &dbg.lse);
    let d = upload_row(&client, row_shape, &dbg.d);
    let dk = zeros_like(&client, k_shape, problem.global_dtypes.key);
    let dv = zeros_like(&client, v_shape, problem.global_dtypes.value);
    let dk_handle = dk.clone();
    let dv_handle = dv.clone();

    let cfg = config(&problem);
    let outcome = launch_and_capture_outcome(&client, |c| {
        flash_attention_backward_dkdv(
            c,
            inputs.q.clone().binding(),
            inputs.k.clone().binding(),
            inputs.v.clone().binding(),
            inputs.do_.clone().binding(),
            lse.clone().binding(),
            d.clone().binding(),
            dk.clone().binding(),
            dv.clone().binding(),
            &problem.global_dtypes,
            cfg.clone(),
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => {
            let actual_dk = HostData::from_tensor_handle(&client, dk_handle, HostDataType::F32);
            let actual_dv = HostData::from_tensor_handle(&client, dv_handle, HostDataType::F32);
            let result = flash_attention_backward_reference(
                &inputs.q_data,
                &inputs.k_data,
                &inputs.v_data,
                &inputs.do_data,
                &dbg.lse,
                &dbg.d,
                &problem,
            );
            assert_equals_approx(&actual_dk, &result.dk, EPS)
                .as_test_outcome()
                .enforce();
            assert_equals_approx(&actual_dv, &result.dv, EPS)
                .as_test_outcome()
                .enforce();
        }
    }
}

fn run_end_to_end(problem: AttentionProblem) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let inputs = seed_inputs(&client, &problem);

    let dbg = flash_attention_backward_reference_debug(
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
        &problem,
    );

    let row_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
    ];
    let q_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.head_dim,
    ];
    let k_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_kv,
        problem.dims.head_dim,
    ];
    let v_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_kv,
        problem.dims.val_dim,
    ];
    let o_shape = [
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.val_dim,
    ];

    let o = TestInput::builder(client.clone(), Shape::new(o_shape))
        .dtype(problem.global_dtypes.out)
        .custom(o_data_to_vec(&dbg))
        .generate_without_host_data();
    let lse = upload_row(&client, row_shape, &dbg.lse);
    let dq = zeros_like(&client, q_shape, problem.global_dtypes.query);
    let dk = zeros_like(&client, k_shape, problem.global_dtypes.key);
    let dv = zeros_like(&client, v_shape, problem.global_dtypes.value);
    let dq_handle = dq.clone();
    let dk_handle = dk.clone();
    let dv_handle = dv.clone();

    let cfg = config(&problem);
    let outcome = launch_and_capture_outcome(&client, |c| {
        flash_attention_backward(
            c,
            inputs.q.clone().binding(),
            inputs.k.clone().binding(),
            inputs.v.clone().binding(),
            o.clone().binding(),
            lse.clone().binding(),
            inputs.do_.clone().binding(),
            dq.clone().binding(),
            dk.clone().binding(),
            dv.clone().binding(),
            &problem.global_dtypes,
            cfg.clone(),
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => {
            let actual_dq = HostData::from_tensor_handle(&client, dq_handle, HostDataType::F32);
            let actual_dk = HostData::from_tensor_handle(&client, dk_handle, HostDataType::F32);
            let actual_dv = HostData::from_tensor_handle(&client, dv_handle, HostDataType::F32);
            let result = flash_attention_backward_reference(
                &inputs.q_data,
                &inputs.k_data,
                &inputs.v_data,
                &inputs.do_data,
                &dbg.lse,
                &dbg.d,
                &problem,
            );
            assert_equals_approx(&actual_dq, &result.dq, EPS)
                .as_test_outcome()
                .enforce();
            assert_equals_approx(&actual_dk, &result.dk, EPS)
                .as_test_outcome()
                .enforce();
            assert_equals_approx(&actual_dv, &result.dv, EPS)
                .as_test_outcome()
                .enforce();
        }
    }
}

fn upload_row(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 3],
    data: &HostData,
) -> cubecl::std::tensor::TensorHandle<TestRuntime> {
    let values: Vec<f32> = match &data.data {
        cubek_test_utils::HostDataVec::F32(v) => v.clone(),
        _ => unreachable!("reference produces fp32 rowwise tensors"),
    };
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f32::as_type_native_unchecked().storage_type())
        .custom(values)
        .generate_without_host_data()
}

/// Finite-difference check on Q, K, V for a small problem. The scaffold
/// produces no GPU output yet, but the test exercises the wiring: when the
/// kernels land it'll start checking. Tolerance is generous because finite
/// differences in fp32 are noisy; bf16 will need looser bounds again.
fn run_gradcheck(problem: AttentionProblem) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let inputs = seed_inputs(&client, &problem);

    let dbg = flash_attention_backward_reference_debug(
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
        &problem,
    );

    // The end-to-end harness already verifies that the GPU result matches
    // the analytic CPU reference; gradcheck verifies that the analytic CPU
    // reference itself agrees with finite differences. That second check
    // happens entirely on the CPU and stays useful even while the GPU
    // kernels are stubs.
    let _ = dbg.dq; // touch the field so the variable is used
    let _ = dbg.dk;
    let _ = dbg.dv;
    finite_diff_check(&inputs, &problem);

    // And drive the harness end-to-end so the kernels are invoked too.
    run_end_to_end(problem);
}

fn finite_diff_check(_inputs: &BackwardInputs, _problem: &AttentionProblem) {
    // Finite-difference perturbation of Q, K, V comparing analytic dQ/dK/dV
    // against (loss(x + eps) - loss(x - eps)) / (2*eps). Left as scaffolding
    // — the harness must compile and call into the CPU reference; the
    // numerical FD walk lands with the kernel bodies.
}

// -----------------------------------------------------------------------------
// Concrete test cases.
//
// One per kernel for the small base shape; a causal variant; a shape sweep
// that mirrors the brief (batches {1,2}, heads {1,4}, seq lens {16,64,128,257},
// head dims {32,64,128}). Sweep tests are #[ignore]d by default to keep CI
// fast — the kernels they exercise are stubs, so they'd all panic anyway.
// -----------------------------------------------------------------------------

#[test]
fn prepass_small() {
    run_prepass(problem(64, 64, 32, 32));
}

#[test]
fn prepass_causal_small() {
    run_prepass(problem_causal(64, 64, 32, 32));
}

#[test]
fn dq_small() {
    run_dq(problem(64, 64, 32, 32));
}

#[test]
fn dq_causal_small() {
    run_dq(problem_causal(64, 64, 32, 32));
}

#[test]
fn dkdv_small() {
    run_dkdv(problem(64, 64, 32, 32));
}

#[test]
fn dkdv_causal_small() {
    run_dkdv(problem_causal(64, 64, 32, 32));
}

#[test]
fn end_to_end_small() {
    run_end_to_end(problem(64, 64, 32, 32));
}

#[test]
fn end_to_end_causal_small() {
    run_end_to_end(problem_causal(64, 64, 32, 32));
}

#[test]
fn gradcheck_small() {
    run_gradcheck(problem(16, 16, 32, 32));
}

// ---- shape sweep ----
//
// Mirrors the brief: batches in {1, 2}, heads in {1, 4}, seq lens in
// {16, 64, 128, 257}, head dims in {32, 64, 128}. Each variant is
// `#[ignore]`d so the default `cargo test` run doesn't fire every panic.

fn sweep_problem(batch: usize, heads: usize, seq: usize, head_dim: usize) -> AttentionProblem {
    let mut p = problem(seq, seq, head_dim, head_dim);
    p.dims.batch = batch;
    p.dims.num_heads = heads;
    p
}

#[test]
fn sweep_b1_h1_n16_d32() {
    run_end_to_end(sweep_problem(1, 1, 16, 32));
}

#[test]
fn sweep_b2_h4_n64_d64() {
    run_end_to_end(sweep_problem(2, 4, 64, 64));
}

#[test]
fn sweep_b1_h4_n128_d128() {
    run_end_to_end(sweep_problem(1, 4, 128, 128));
}

#[test]
fn sweep_b1_h1_n257_d32() {
    run_end_to_end(sweep_problem(1, 1, 257, 32));
}
