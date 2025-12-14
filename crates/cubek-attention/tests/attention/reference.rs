use cubek_attention::launch::{AttentionDefinition, AttentionElems};

use core::f32;

use cubecl::{TestRuntime, client::ComputeClient, std::tensor::TensorHandle};

use cubek_std::test_utils::{
    HostData, HostDataType, HostDataVec, StrideSpec, assert_equals_approx,
};

pub fn assert_result(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    mask: Option<&HostData>,
    definition: &AttentionDefinition,
    client: &ComputeClient<TestRuntime>,
    out: TensorHandle<TestRuntime>,
    elems: AttentionElems,
) {
    let epsilon = attention_epsilon(&elems, 0.1);
    let expected = flash_attention_v2_reference(query, key, value, mask, definition);

    let actual = HostData::from_tensor_handle(client, &out, HostDataType::F32);

    if let Err(e) = assert_equals_approx(&actual, &expected, epsilon) {
        panic!("{}", e);
    }
}

fn attention_epsilon(elems: &AttentionElems, safety_factor: f32) -> f32 {
    let total_eps = [
        elems.query_global.epsilon(),
        elems.query_tile.epsilon(),
        elems.key_global.epsilon(),
        elems.key_stage.epsilon(),
        elems.value_global.epsilon(),
        elems.value_stage.epsilon(),
        elems.key_value_tile.epsilon(),
        elems.softmax.epsilon(),
        elems.accumulator.epsilon(),
        elems.mask.epsilon(),
        elems.out_global.epsilon(),
        elems.out_stage.epsilon(),
    ]
    .into_iter()
    .fold(0.0, f64::max);

    total_eps as f32 * safety_factor
}
pub fn flash_attention_v2_reference(
    query: &HostData,
    key: &HostData,
    value: &HostData,
    mask: Option<&HostData>,
    definition: &AttentionDefinition,
) -> HostData {
    let batch = definition.dims.batch;
    let seq_q = definition.dims.seq_q;
    let seq_kv = definition.dims.seq_kv;
    let num_heads = definition.dims.num_heads;
    let head_dim = definition.dims.head_dim;
    let val_dim = definition.dims.val_dim;

    let masked = mask.is_some();
    assert!(definition.masked == masked);

    // Output shape: [batch, num_heads, seq_q, val_dim]
    let out_shape = vec![batch, num_heads, seq_q, val_dim];
    let mut out = vec![0.; batch * num_heads * seq_q * val_dim];

    let scale = (head_dim as f32).sqrt().recip();

    // Use fixed-size arrays instead of heap Vec
    let mut q_index: [usize; 4];
    let mut k_index: [usize; 4];
    let mut v_index: [usize; 4];
    let mut m_index: [usize; 4];
    let mut out_index = vec![0usize; 4];

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_q {
                // initialize running row accumulator
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.;
                let mut acc_row = vec![0.; val_dim];

                for j in 0..seq_kv {
                    // compute dot(Q_i, K_j)
                    let mut dot = 0.;
                    for d in 0..head_dim {
                        q_index = [b, h, i, d];
                        k_index = [b, h, j, d];
                        dot += query.get_f32(&q_index) * key.get_f32(&k_index);
                    }
                    dot *= scale;

                    // apply causal/external mask
                    let s_val = if definition.options.causal && j > i {
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

                    // skip update if row is fully masked (prevent NaNs)
                    if s_val == f32::NEG_INFINITY && m == f32::NEG_INFINITY {
                        continue;
                    }

                    // update row max
                    let m_new = m.max(s_val);

                    // compute exp(S - m_new)
                    let p_tilde = f32::exp(s_val - m_new);

                    // update running sum l
                    let l_new = f32::exp(m - m_new) * l + p_tilde;

                    // update accumulator: acc = exp(m - m_new) * acc + p_tilde * V_j
                    let scale_old = f32::exp(m - m_new);
                    for d in 0..val_dim {
                        acc_row[d] *= scale_old;
                        v_index = [b, h, j, d];
                        acc_row[d] += p_tilde * value.get_f32(&v_index);
                    }

                    // commit
                    m = m_new;
                    l = l_new;
                }

                // normalize and write output
                out_index[0] = b;
                out_index[1] = h;
                out_index[2] = i;
                let eps = 1e-20f32; // numerical safety
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
