use cubek_attention::launch::{AttentionDefinition, AttentionElems, AttentionIdent};

use core::f32;

use cubecl::{TestRuntime, client::ComputeClient, std::tensor::TensorHandle};

use cubek_std::test_utils::assert_equals_approx;

pub fn assert_result(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<&[bool]>,
    definition: &AttentionDefinition,
    client: &ComputeClient<TestRuntime>,
    out: TensorHandle<TestRuntime>,
    elems: AttentionElems,
) {
    let epsilon = attention_epsilon(&elems, 170.);
    let expected = flash_attention_v2_reference(query, key, value, mask, definition);

    if let Err(e) = assert_equals_approx(client, &out, &expected, epsilon) {
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

pub(crate) fn flash_attention_v2_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<&[bool]>,
    definition: &AttentionDefinition,
) -> Vec<f32>
where
{
    let batch = definition.dims.batch;
    let seq_q = definition.dims.seq_q;
    let seq_kv = definition.dims.seq_kv;
    let num_heads = definition.dims.num_heads;
    let head_dim = definition.dims.head_dim;
    let val_dim = definition.dims.val_dim;

    let masked = mask.is_some();
    assert!(definition.masked == masked);

    // Precompute strides for indexing
    let query_strides = strides(definition, AttentionIdent::Query);
    let key_strides = strides(definition, AttentionIdent::Key);
    let value_strides = strides(definition, AttentionIdent::Value);
    let mask_strides = strides(definition, AttentionIdent::Mask);
    let out_strides = strides(definition, AttentionIdent::Out);

    let out_size = definition.shape(AttentionIdent::Out).iter().product();
    let mut out = vec![0.; out_size];

    // scaling factor 1/sqrt(head_dim)
    let scale = (head_dim as f32).sqrt().recip();

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_q {
                // Initialize running state for query row i
                // m = -inf, l = 0, accumulator O (unnormalized numerator) = 0
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.;
                let mut acc_row = vec![0.; val_dim];

                // For each K/V block
                let mut k_block_start = 0usize;
                while k_block_start < seq_kv {
                    let k_block_end = std::cmp::min(seq_kv, k_block_start + seq_kv);
                    let cur_block_len = k_block_end - k_block_start;

                    // Step A: compute S_block[j'] = Q_i · K_{j'}  for j' in block
                    // store in a small Vec<P::EA>
                    let mut s_block = vec![0.; cur_block_len];
                    for (bj, j) in (k_block_start..k_block_end).enumerate() {
                        let mut dot = 0.;
                        for d in 0..head_dim {
                            let q_idx = b * query_strides[0]
                                + h * query_strides[1]
                                + i * query_strides[2]
                                + d * query_strides[3];
                            let k_idx = b * key_strides[0]
                                + h * key_strides[1]
                                + j * key_strides[2]
                                + d * key_strides[3];
                            let q_val = query[q_idx];
                            let k_val = key[k_idx];

                            dot += q_val * k_val;
                        }
                        // apply scale (1/sqrt(head_dim))
                        dot *= scale;

                        // Apply mask if applicable
                        let s_val = if definition.options.causal && j > i {
                            // Causal mask
                            f32::NEG_INFINITY
                        } else if masked {
                            // Explicit mask
                            let m_idx = b * mask_strides[0]
                                + h * mask_strides[1]
                                + i * mask_strides[2]
                                + j * mask_strides[3];
                            let m_val = mask.unwrap()[m_idx];

                            if m_val { f32::NEG_INFINITY } else { dot }
                        } else {
                            dot
                        };

                        s_block[bj] = s_val;
                    }

                    // Step B: compute new row max m' = max(m, rowmax(S_block))
                    let mut block_max = f32::NEG_INFINITY;
                    for &v_s in &s_block {
                        if v_s > block_max {
                            block_max = v_s;
                        }
                    }

                    if block_max == f32::NEG_INFINITY {
                        // the numerator is zero, so simply keep m, l, acc_row unchanged.
                        // Move to next block.
                        k_block_start += cur_block_len;
                        continue;
                    }

                    // m_new
                    let mut m_new = m;
                    if block_max > m_new {
                        m_new = block_max;
                    }

                    // Step C: compute Ptilde = exp(S_block - m_new)
                    // and rowsum = sum Ptilde
                    let mut rowsum = 0.;
                    let mut p_tilde = vec![0.; cur_block_len];
                    for (bj, &sval) in s_block.iter().enumerate() {
                        let e = f32::exp(sval - m_new);

                        p_tilde[bj] = e;
                        rowsum += e;
                    }

                    // Step D: update running l: l_new = exp(m - m_new)*l + rowsum
                    // note: exp(prev_m - m_new) where prev_m==m
                    let epm = f32::exp(m - m_new);
                    let l_new = epm * l + rowsum;

                    // Step E: update numerator accumulator:
                    // acc = exp(m - m_new) * acc + Ptilde @ V_block
                    // First scale old accumulator by epm
                    for d in 0..val_dim {
                        acc_row[d] = epm * acc_row[d];
                    }
                    // Add Ptilde @ V_block
                    for (bj, j) in (k_block_start..k_block_end).enumerate() {
                        let p_val = p_tilde[bj];
                        for d in 0..val_dim {
                            let v_idx = b * value_strides[0]
                                + h * value_strides[1]
                                + j * value_strides[2]
                                + d * value_strides[3];
                            let v_val = value[v_idx];
                            acc_row[d] += p_val * v_val;
                        }
                    }

                    // commit updated m and l for next block
                    m = m_new;
                    l = l_new;

                    // next block
                    k_block_start += cur_block_len;
                } // end while over K/V blocks

                // Step final: normalize accumulator: O_final = acc_row / l
                // write into output
                let out_base = b * out_strides[0] + h * out_strides[1] + i * out_strides[2];

                // guard against tiny l (numerical safety)
                let eps = 1e-20f32;
                let denom = if l > eps { l } else { eps };
                for d in 0..val_dim {
                    let out_idx = out_base + d * out_strides[3];
                    out[out_idx] = acc_row[d] / denom;
                }
            }
        }
    }

    out
}

pub(crate) fn strides(problem: &AttentionDefinition, ident: AttentionIdent) -> Vec<usize> {
    let shape = problem.shape(ident);

    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }

    strides
}
