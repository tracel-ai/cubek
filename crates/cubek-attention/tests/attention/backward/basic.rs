//! Backward-pass test harness.
//!
//! Mirrors `basic.rs`: each test builds a seeded input set, routes it
//! through one of the three backward kernels (or the end-to-end entry
//! point), and compares the GPU output against the CPU reference in
//! [`cubek_attention::eval::backward::cpu_reference`].
//!
//! The kernels are implemented (per-row loops, fp32 accumulation, generic
//! over the f32/f16/bf16 I/O dtype) and these tests prove them on CUDA
//! against the CPU reference: small shapes in f32, pythia-160m-like shapes
//! (causal, head_dim 64, seq 128 and 1024) in f32 and bf16, and a
//! finite-difference check of the CPU oracle itself.

#![cfg(feature = "cpu-reference")]

use cubecl::{client::Client, prelude::Scalar, zspace::Shape};
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

/// Seeded inputs + their fp32 host-side mirrors. The device tensors use the
/// problem's dtype; the host mirrors are read back from them, so they hold the
/// already-rounded values the kernels actually see (matters for bf16).
struct BackwardInputs {
    q: cubecl::std::tensor::TensorHandle,
    q_data: HostData,
    k: cubecl::std::tensor::TensorHandle,
    k_data: HostData,
    v: cubecl::std::tensor::TensorHandle,
    v_data: HostData,
    do_: cubecl::std::tensor::TensorHandle,
    do_data: HostData,
}

fn problem(seq_q: usize, seq_kv: usize, head_dim: usize, val_dim: usize) -> AttentionProblem {
    let client = cubecl::test_device().client();
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
            f32::elem_type_native(),
            AttentionGlobalTypes::mask_dtype(&client),
        ),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: Default::default(),
    }
}

/// Same problem with a different shared float dtype for q/k/v/out.
fn with_dtype(mut p: AttentionProblem, dtype: cubecl::ir::ElemType) -> AttentionProblem {
    let client = cubecl::test_device().client();
    p.global_dtypes = AttentionGlobalTypes::from_single_float_dtype(
        dtype,
        AttentionGlobalTypes::mask_dtype(&client),
    );
    p
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

fn seed_inputs(client: &Client, problem: &AttentionProblem) -> BackwardInputs {
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
    client: &Client,
    shape: [usize; 4],
    dtype: cubecl::ir::ElemType,
) -> cubecl::std::tensor::TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

fn zeros_row(
    client: &Client,
    shape: [usize; 3],
    dtype: cubecl::ir::ElemType,
) -> cubecl::std::tensor::TensorHandle {
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

/// Absolute epsilon for the small f32 shapes (gradients are O(1e-2..1e-1)).
const EPS: f32 = 1e-3;

/// Error budget for the large-shape tests, relative to `max |reference|` of the
/// tensor being checked (gradient magnitudes differ a lot between dQ and dV).
#[derive(Clone, Copy)]
enum Tol {
    /// Absolute, via `assert_equals_approx`.
    Abs(f32),
    /// Fraction of `max |expected|`, with the measured errors printed.
    Rel(f32),
}

/// f32: sequential accumulation over <= 1024 terms is bounded by
/// 1024 * 2^-24 = 6.1e-5 of the magnitude; 1e-4 leaves headroom for the GPU's
/// approximate `exp` and different summation order.
const F32_REL: f32 = 1e-4;

/// bf16 has 8 significand bits, so the unit roundoff is u = 2^-8 (relative
/// error of one rounding is up to u, not 2^-9). Two roundings reach the result:
/// the bf16 output itself (<= u of the element, hence <= u of the max), and the
/// bf16 `O` feeding `D = rowsum(dO * O)`, whose error propagates into `dS` at
/// about the same relative size. All inner products accumulate in fp32, so the
/// accumulation length adds nothing. Budget: 2u = 2^-7 of `max |ref|`.
const BF16_REL: f32 = 1.0 / 128.0;

/// Compare one gradient tensor and print max abs / relative errors.
fn check(name: &str, actual: &HostData, expected: &HostData, tol: Tol) {
    match tol {
        Tol::Abs(eps) => assert_equals_approx(actual, expected, eps)
            .as_test_outcome()
            .enforce(),
        Tol::Rel(rel) => {
            assert_eq!(actual.shape, expected.shape, "{name}: shape mismatch");
            let n: usize = expected.shape.iter().product();
            let (mut max_abs, mut max_ref, mut sum_abs) = (0f32, 0f32, 0f64);
            for i in 0..n {
                let (a, e) = (actual.data.get_f32(i), expected.data.get_f32(i));
                let err = (a - e).abs();
                assert!(err.is_finite(), "{name}[{i}]: got {a}, expected {e}");
                max_abs = max_abs.max(err);
                max_ref = max_ref.max(e.abs());
                sum_abs += err as f64;
            }
            println!(
                "{name}: max_abs_err={max_abs:.3e} max_ref={max_ref:.3e} rel={:.3e} mean_abs_err={:.3e} (budget rel={rel:.3e})",
                max_abs / max_ref,
                sum_abs / n as f64
            );
            assert!(
                max_abs <= rel * max_ref,
                "{name}: max abs err {max_abs:.3e} exceeds {rel:.3e} * max|ref| {max_ref:.3e}"
            );
        }
    }
}

/// Run the prepass kernel and compare its `D` output against the CPU
/// reference.
fn run_prepass(problem: AttentionProblem) {
    let client = cubecl::test_device().client();
    let inputs = seed_inputs(&client, &problem);

    // We need O for the prepass: generate it by running the CPU forward.
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

    let d = zeros_row(&client, row_shape, f32::elem_type_native());
    let d_handle = d.clone();

    let outcome = launch_and_capture_outcome(&client, &[&d.handle], |c| {
        flash_attention_backward_prepass(
            c,
            o.clone().binding(),
            inputs.do_.clone().binding(),
            d.clone().binding(),
            &problem.global_dtypes,
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
/// against the CPU reference's dQ.
fn run_dq(problem: AttentionProblem) {
    let client = cubecl::test_device().client();
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
    let outcome = launch_and_capture_outcome(&client, &[&dq.handle], |c| {
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
    let client = cubecl::test_device().client();
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
    let outcome = launch_and_capture_outcome(&client, &[&dk.handle, &dv.handle], |c| {
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
    run_end_to_end_tol(problem, Tol::Abs(EPS));
}

fn run_end_to_end_tol(problem: AttentionProblem, tol: Tol) {
    let client = cubecl::test_device().client();
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
    let outcome = launch_and_capture_outcome(&client, &[&dq.handle, &dk.handle, &dv.handle], |c| {
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
            // The debug reference derives lse and D from its own forward, which
            // is exactly what was uploaded above, so its gradients are the oracle.
            check("dQ", &actual_dq, &dbg.dq, tol);
            check("dK", &actual_dk, &dbg.dk, tol);
            check("dV", &actual_dv, &dbg.dv, tol);
        }
    }
}

fn upload_row(
    client: &Client,
    shape: [usize; 3],
    data: &HostData,
) -> cubecl::std::tensor::TensorHandle {
    let values: Vec<f32> = match &data.data {
        cubek_test_utils::HostDataVec::F32(v) => v.clone(),
        _ => unreachable!("reference produces fp32 rowwise tensors"),
    };
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f32::elem_type_native())
        .custom(values)
        .generate_without_host_data()
}

/// Finite-difference check of the CPU oracle on Q, K, V for a small problem,
/// then the GPU end-to-end comparison. The analytic reference is only trusted as
/// ground truth after agreeing with `(L(x + h) - L(x - h)) / 2h`, where
/// `L = sum(dO * O)` so that `dL/dx` equals the backward output for `x`.
fn run_gradcheck(problem: AttentionProblem) {
    let client = cubecl::test_device().client();
    let inputs = seed_inputs(&client, &problem);

    let dbg = flash_attention_backward_reference_debug(
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
        &problem,
    );
    finite_diff_check(&inputs, &problem, &dbg);

    run_end_to_end(problem);
}

fn f32_vec(h: &HostData) -> Vec<f32> {
    match &h.data {
        cubek_test_utils::HostDataVec::F32(v) => v.clone(),
        _ => unreachable!("host mirrors are fp32"),
    }
}

/// `x` with element `i` shifted by `delta`.
fn shifted(x: &HostData, i: usize, delta: f32) -> HostData {
    let mut v = f32_vec(x);
    v[i] += delta;
    HostData {
        data: cubek_test_utils::HostDataVec::F32(v),
        shape: x.shape.clone(),
        strides: x.strides.clone(),
    }
}

/// `L = sum(dO * O(q, k, v))` through the CPU forward.
fn loss(q: &HostData, k: &HostData, v: &HostData, do_: &HostData, p: &AttentionProblem) -> f64 {
    let o = f32_vec(&flash_attention_backward_reference_debug(q, k, v, do_, p).o);
    let g = f32_vec(do_);
    o.iter().zip(&g).map(|(a, b)| *a as f64 * *b as f64).sum()
}

fn finite_diff_check(
    inputs: &BackwardInputs,
    problem: &AttentionProblem,
    dbg: &FlashAttentionBackwardDebug,
) {
    // h = 1e-2: truncation error O(h^2 * third derivative) stays below the f32 rounding noise
    // of L (~1e-7 * |L| / h) for these O(1) inputs.
    const H: f32 = 1e-2;
    const TOL: f64 = 2e-3;
    let (q, k, v, g) = (
        &inputs.q_data,
        &inputs.k_data,
        &inputs.v_data,
        &inputs.do_data,
    );
    for (name, x, analytic) in [("q", q, &dbg.dq), ("k", k, &dbg.dk), ("v", v, &dbg.dv)] {
        let n: usize = x.shape.iter().product();
        let grads = f32_vec(analytic);
        // Seven spread-out elements per tensor, including both ends.
        for t in 0..7 {
            let i = t * (n - 1) / 6;
            let eval = |delta: f32| {
                let xs = shifted(x, i, delta);
                match name {
                    "q" => loss(&xs, k, v, g, problem),
                    "k" => loss(q, &xs, v, g, problem),
                    _ => loss(q, k, &xs, g, problem),
                }
            };
            let fd = (eval(H) - eval(-H)) / (2.0 * H as f64);
            let err = (fd - grads[i] as f64).abs();
            assert!(
                err <= TOL,
                "d{name}[{i}]: finite difference {fd:.6} vs analytic {:.6} (err {err:.2e})",
                grads[i]
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Concrete test cases.
//
// One per kernel for the small base shape; a causal variant; a shape sweep
// that mirrors the brief (batches {1,2}, heads {1,4}, seq lens {16,64,128,257},
// head dims {32,64,128}), and the pythia-160m-like cases below.
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
// {16, 64, 128, 257}, head dims in {32, 64, 128}.

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

// ---- pythia-160m-like shapes: head_dim 64, multi-batch / multi-head, seq_q == seq_kv ----

fn model_problem(seq: usize, causal: bool, dtype: cubecl::ir::ElemType) -> AttentionProblem {
    let mut p = sweep_problem(2, 4, seq, 64);
    p.options.causal = causal;
    with_dtype(p, dtype)
}

#[test]
fn f32_b2_h4_n128_d64() {
    run_end_to_end_tol(
        model_problem(128, false, f32::elem_type_native()),
        Tol::Rel(F32_REL),
    );
}

#[test]
fn f32_causal_b2_h4_n128_d64() {
    run_end_to_end_tol(
        model_problem(128, true, f32::elem_type_native()),
        Tol::Rel(F32_REL),
    );
}

#[test]
fn f32_causal_b2_h4_n1024_d64() {
    run_end_to_end_tol(
        model_problem(1024, true, f32::elem_type_native()),
        Tol::Rel(F32_REL),
    );
}

#[test]
fn bf16_causal_b2_h4_n128_d64() {
    run_end_to_end_tol(
        model_problem(128, true, half::bf16::elem_type_native()),
        Tol::Rel(BF16_REL),
    );
}

#[test]
fn bf16_causal_b2_h4_n1024_d64() {
    run_end_to_end_tol(
        model_problem(1024, true, half::bf16::elem_type_native()),
        Tol::Rel(BF16_REL),
    );
}

/// Mixed q/k/v/out dtypes must be a typed error, not a silent reinterpretation.
#[test]
fn mixed_dtypes_rejected() {
    let client = cubecl::test_device().client();
    let mut p = model_problem(16, true, half::bf16::elem_type_native());
    p.global_dtypes.out = f32::elem_type_native();
    let inputs = seed_inputs(&client, &p);
    let row = zeros_row(&client, [2, 4, 16], f32::elem_type_native());
    let r = flash_attention_backward_prepass(
        &client,
        inputs.q.clone().binding(),
        inputs.do_.clone().binding(),
        row.clone().binding(),
        &p.global_dtypes,
    );
    assert!(r.is_err());
}

/// Wall time of the full backward at the pythia-160m training shape (b4, h12,
/// seq 1024, d64, bf16, causal). Run with `-- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_bf16_causal_b4_h12_n1024_d64() {
    let client = cubecl::test_device().client();
    let mut p = sweep_problem(4, 12, 1024, 64);
    p.options.causal = true;
    let p = with_dtype(p, half::bf16::elem_type_native());
    let inputs = seed_inputs(&client, &p);
    let row_shape = [4, 12, 1024];
    let o_shape = [4, 12, 1024, 64];
    let o = TestInput::builder(client.clone(), Shape::new(o_shape))
        .dtype(p.global_dtypes.out)
        .uniform(55, -1., 1.)
        .generate_without_host_data();
    let lse = TestInput::builder(client.clone(), Shape::new(row_shape))
        .dtype(f32::elem_type_native())
        .uniform(66, 4., 7.)
        .generate_without_host_data();
    let dq = zeros_like(&client, o_shape, p.global_dtypes.query);
    let dk = zeros_like(&client, o_shape, p.global_dtypes.key);
    let dv = zeros_like(&client, o_shape, p.global_dtypes.value);
    let cfg = config(&p);

    let run = || {
        flash_attention_backward(
            &client,
            inputs.q.clone().binding(),
            inputs.k.clone().binding(),
            inputs.v.clone().binding(),
            o.clone().binding(),
            lse.clone().binding(),
            inputs.do_.clone().binding(),
            dq.clone().binding(),
            dk.clone().binding(),
            dv.clone().binding(),
            &p.global_dtypes,
            cfg.clone(),
        )
        .unwrap();
        cubecl::future::block_on(client.sync()).unwrap();
    };
    for _ in 0..2 {
        run();
    }
    let mut ms: Vec<f64> = (0..7)
        .map(|_| {
            let t = std::time::Instant::now();
            run();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(|a, b| a.total_cmp(b));
    println!(
        "flash_attention_backward b4 h12 n1024 d64 bf16 causal: median {:.2} ms, min {:.2} ms, max {:.2} ms",
        ms[3], ms[0], ms[6]
    );
}
