//! CPU reference and seeded "produce a HostData" primitives for attention.
//!
//! - [`strategy_result`] runs the kernel once and returns its output as a
//!   [`HostData`].
//! - [`cpu_reference_result`] runs the naive flash-attention-v2 reference on
//!   the same seeded inputs and returns its output as a [`HostData`].

use core::f32;

use cubecl::{TestRuntime, client::ComputeClient, std::tensor::TensorHandle, zspace::Shape};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput, ValidationResult,
    assert_equals_approx, launch_and_capture_outcome,
};

use crate::{
    definition::{AttentionElems, AttentionIdent, AttentionOptions, AttentionProblem},
    launch::{Strategy, launch_ref},
};

/// Run `strategy` against `problem` with seeded inputs and return its output as
/// a [`HostData`].
///
/// Inputs are generated via `TestInput::uniform`/`bernoulli` so the same
/// `(problem, seeds)` pair produces the same bits on every run.
pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    strategy: Strategy,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let inputs = seed_inputs(&client, &problem, seed_lhs, seed_rhs);
    let out_handle = inputs.out.clone();
    let mask_binding = inputs.mask.as_ref().map(|m| m.clone().binding());

    let outcome = launch_and_capture_outcome(&client, |c| {
        launch_ref(
            strategy.clone(),
            c,
            inputs.query.clone().binding(),
            inputs.key.clone().binding(),
            inputs.value.clone().binding(),
            mask_binding.clone(),
            out_handle.clone().binding(),
            &problem.global_dtypes,
            AttentionOptions {
                causal: problem.options.causal,
                accumulator_precision: problem.options.accumulator_precision.clone(),
            },
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            out_handle,
            HostDataType::F32,
        )),
    }
}

/// CPU-only counterpart to [`strategy_result`]: generate the same seeded
/// inputs, run the naive flash-attention-v2 reference, return the result as
/// a [`HostData`].
///
/// Slow on bench-scale problems by design — only useful as a ground truth.
pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let inputs = seed_inputs(&client, &problem, seed_lhs, seed_rhs);
    Ok(flash_attention_v2_reference(
        &inputs.query_data,
        &inputs.key_data,
        &inputs.value_data,
        inputs.mask_data.as_ref(),
        &problem,
    ))
}

struct SeededInputs {
    query: TensorHandle<TestRuntime>,
    query_data: HostData,
    key: TensorHandle<TestRuntime>,
    key_data: HostData,
    value: TensorHandle<TestRuntime>,
    value_data: HostData,
    mask: Option<TensorHandle<TestRuntime>>,
    mask_data: Option<HostData>,
    out: TensorHandle<TestRuntime>,
}

fn seed_inputs(
    client: &ComputeClient<TestRuntime>,
    problem: &AttentionProblem,
    seed_lhs: u64,
    seed_rhs: u64,
) -> SeededInputs {
    // Two 64-bit seeds need to fan out into four; mix them deterministically so
    // the same `(seed_lhs, seed_rhs)` pair always reproduces the same inputs.
    let seed_value = seed_lhs.wrapping_add(seed_rhs).wrapping_add(0x9e37_79b9);
    let seed_mask = seed_lhs
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(seed_rhs);

    let (query, query_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Query)),
    )
    .dtype(problem.global_dtypes.query)
    .uniform(seed_lhs, -1., 1.)
    .generate_with_f32_host_data();

    let (key, key_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Key)),
    )
    .dtype(problem.global_dtypes.key)
    .uniform(seed_rhs, -1., 1.)
    .generate_with_f32_host_data();

    let (value, value_data) = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Value)),
    )
    .dtype(problem.global_dtypes.value)
    .uniform(seed_value, -1., 1.)
    .generate_with_f32_host_data();

    let (mask, mask_data) = if problem.masked {
        let (m, d) = TestInput::builder(
            client.clone(),
            Shape::new(problem.shape(AttentionIdent::Mask)),
        )
        .dtype(problem.global_dtypes.mask)
        .bernoulli(seed_mask, 0.1)
        .generate_with_bool_host_data();
        (Some(m), Some(d))
    } else {
        (None, None)
    };

    let out = TestInput::builder(
        client.clone(),
        Shape::new(problem.shape(AttentionIdent::Out)),
    )
    .dtype(problem.global_dtypes.out)
    .zeros()
    .generate_without_host_data();

    SeededInputs {
        query,
        query_data,
        key,
        key_data,
        value,
        value_data,
        mask,
        mask_data,
        out,
    }
}

/// Mirror of [`assert_equals_approx`] specialized to attention; same epsilon
/// rules as the existing test helper.
#[allow(clippy::too_many_arguments)]
pub fn assert_result(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    mask: Option<&HostData>,
    problem: &AttentionProblem,
    client: &ComputeClient<TestRuntime>,
    out: TensorHandle<TestRuntime>,
    elems: AttentionElems,
) -> ValidationResult {
    let epsilon = attention_epsilon(&elems, 0.01);
    let expected = flash_attention_v2_reference(query, key, value, mask, problem);
    let actual = HostData::from_tensor_handle(client, out, HostDataType::F32);

    assert_equals_approx(&actual, &expected, epsilon)
}

/// Default attention-side epsilon × safety factor.
pub fn attention_epsilon(elems: &AttentionElems, safety_factor: f32) -> f32 {
    let total_eps = [
        elems.query_global.epsilon(),
        elems.query_tile.epsilon(),
        elems.key_global.epsilon(),
        elems.key_stage.epsilon(),
        elems.value_global.epsilon(),
        elems.value_stage.epsilon(),
        elems.key_value_tile.epsilon(),
        elems.softmax_acc.epsilon(),
        elems.accumulator.epsilon(),
        elems.mask.epsilon(),
        elems.out_global.epsilon(),
        elems.out_stage.epsilon(),
    ]
    .into_iter()
    .fold(0.0, f64::max);

    total_eps as f32 * safety_factor
}

/// Naive flash-attention-v2 reference. Slow on large payloads — intended only
/// for testing.
pub fn flash_attention_v2_reference(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    mask: Option<&HostData>,
    problem: &AttentionProblem,
) -> HostData {
    let batch = problem.dims.batch;
    let seq_q = problem.dims.seq_q;
    let seq_kv = problem.dims.seq_kv;
    let num_heads = problem.dims.num_heads;
    let head_dim = problem.dims.head_dim;
    let val_dim = problem.dims.val_dim;

    let masked = mask.is_some();
    assert!(problem.masked == masked);

    let out_shape = Shape::new([batch, num_heads, seq_q, val_dim]);
    let mut out = vec![0.; batch * num_heads * seq_q * val_dim];

    let scale = (head_dim as f32).sqrt().recip();

    let mut q_index: [usize; 4];
    let mut k_index: [usize; 4];
    let mut v_index: [usize; 4];
    let mut m_index: [usize; 4];
    let mut out_index = [0usize; 4];

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_q {
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.;
                let mut acc_row = vec![0.; val_dim];

                for j in 0..seq_kv {
                    let mut dot = 0.;
                    for d in 0..head_dim {
                        q_index = [b, h, i, d];
                        k_index = [b, h, j, d];
                        dot += query.get_f32(&q_index) * key.get_f32(&k_index);
                    }
                    dot *= scale;

                    let s_val = if problem.options.causal && j > i {
                        f32::NEG_INFINITY
                    } else if let Some(mask) = mask {
                        m_index = [b, h, i, j];
                        if mask.get_bool(&m_index) {
                            f32::NEG_INFINITY
                        } else {
                            dot
                        }
                    } else {
                        dot
                    };

                    if s_val == f32::NEG_INFINITY && m == f32::NEG_INFINITY {
                        continue;
                    }

                    let m_new = m.max(s_val);
                    let p_tilde = f32::exp(s_val - m_new);
                    let l_new = f32::exp(m - m_new) * l + p_tilde;

                    let scale_old = f32::exp(m - m_new);
                    for d in 0..val_dim {
                        acc_row[d] *= scale_old;
                        v_index = [b, h, j, d];
                        acc_row[d] += p_tilde * value.get_f32(&v_index);
                    }

                    m = m_new;
                    l = l_new;
                }

                out_index[0] = b;
                out_index[1] = h;
                out_index[2] = i;
                let eps = 1e-20f32;
                let denom = if l > eps { l } else { eps };
                for d in 0..val_dim {
                    out_index[3] = d;
                    let linear_idx = out_index[0] * num_heads * seq_q * val_dim
                        + out_index[1] * seq_q * val_dim
                        + out_index[2] * val_dim
                        + d;
                    out[linear_idx] = acc_row[d] / denom;
                }
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
