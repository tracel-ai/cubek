//! Kernel 2: compute `dQ`. Q-outer loop, accumulates in registers, no atomics.
//!
//! Each program instance owns one Q-block and sweeps all KV-blocks. `S` and
//! `dP` are recomputed inside the loop, never materialized to HBM.

use cubecl::{CubeDim, calculate_cube_count_elemwise, client::Client, prelude::*};

use super::io_dtype;
use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dq_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dq: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = ABSOLUTE_POS;
    if row_idx >= lse.len() {
        terminate!();
    }

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let i = row_idx % seq_q;
    let bh = row_idx / seq_q;

    let q_base = row_idx * head_dim;
    let do_base = row_idx * val_dim;
    let dq_base = row_idx * head_dim;
    let k_row_base = bh * seq_kv * head_dim;
    let v_row_base = bh * seq_kv * val_dim;

    let lse_i = lse[row_idx];
    let d_i = d[row_idx];

    let mut dq_acc = Array::<f32>::new(head_dim);
    for dd in 0..head_dim {
        dq_acc[dd] = 0.0_f32;
    }

    for j in 0..seq_kv {
        let masked = causal && j + seq_q > i + seq_kv;
        if !masked {
            let k_base = k_row_base + j * head_dim;
            let v_base = v_row_base + j * val_dim;

            let mut dot = 0.0_f32;
            for dd in 0..head_dim {
                dot += f32::cast_from(q[q_base + dd]) * f32::cast_from(k[k_base + dd]);
            }
            let s_ij = dot * scale;
            let p_ij = (s_ij - lse_i).exp();

            let mut dp = 0.0_f32;
            for dd in 0..val_dim {
                dp += f32::cast_from(do_[do_base + dd]) * f32::cast_from(v[v_base + dd]);
            }

            let s_ds = scale * p_ij * (dp - d_i);

            for dd in 0..head_dim {
                dq_acc[dd] += s_ds * f32::cast_from(k[k_base + dd]);
            }
        }
    }

    for dd in 0..head_dim {
        dq[dq_base + dd] = E::cast_from(dq_acc[dd]);
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dq_tiled_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dq: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let c_d = comptime!(head_dim.div_ceil(32));
    let c_v = comptime!(val_dim.div_ceil(32));
    let k_sh_len = comptime!(32 * head_dim);
    let v_sh_len = comptime!(32 * val_dim);

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let total_q_rows = lse.len();
    let total_bh = total_q_rows / seq_q;
    let q_tiles_per_bh = seq_q.div_ceil(4);

    let cube_id = CUBE_POS;
    let bh = cube_id / q_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let q_tile_idx = cube_id % q_tiles_per_bh;
    let i_base = q_tile_idx * 4;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;

    let i = i_base + w;
    let valid_i = i < seq_q;
    let row_idx = bh * seq_q + i;

    let q_base = row_idx * head_dim;
    let do_base = row_idx * val_dim;
    let dq_base = row_idx * head_dim;
    let k_bh_base = bh * seq_kv * head_dim;
    let v_bh_base = bh * seq_kv * val_dim;

    let mut q_reg = Array::<f32>::new(c_d);
    let mut do_reg = Array::<f32>::new(c_v);
    let mut dq_acc = Array::<f32>::new(c_d);

    #[unroll]
    for c in 0..c_d {
        let dd = c * 32 + lane;
        if valid_i && dd < head_dim {
            q_reg[c] = f32::cast_from(q[q_base + dd]);
        } else {
            q_reg[c] = 0.0_f32;
        }
        dq_acc[c] = 0.0_f32;
    }

    #[unroll]
    for c in 0..c_v {
        let dd = c * 32 + lane;
        if valid_i && dd < val_dim {
            do_reg[c] = f32::cast_from(do_[do_base + dd]);
        } else {
            do_reg[c] = 0.0_f32;
        }
    }

    let mut lse_i = 0.0_f32;
    let mut d_i = 0.0_f32;
    if valid_i {
        lse_i = lse[row_idx];
        d_i = d[row_idx];
    }

    let i_end = if seq_q < i_base + 4 {
        seq_q
    } else {
        i_base + 4
    };
    let i_max = i_end - 1;
    let max_kv = if causal {
        if i_max + seq_kv < seq_q {
            0usize
        } else {
            let lim = i_max + seq_kv + 1 - seq_q;
            if seq_kv < lim { seq_kv } else { lim }
        }
    } else {
        seq_kv
    };

    let mut k_sh = Shared::<[f32]>::new_slice(k_sh_len);
    let mut v_sh = Shared::<[f32]>::new_slice(v_sh_len);

    let mut j_tile = 0usize;
    while j_tile < max_kv {
        let rem = max_kv - j_tile;
        let tile_rows = if rem < 32usize { rem } else { 32usize };

        let k_tile_base = k_bh_base + j_tile * head_dim;
        let k_elems = tile_rows * head_dim;
        let mut idx = tid;
        while idx < k_elems {
            k_sh[idx] = f32::cast_from(k[k_tile_base + idx]);
            idx += 128;
        }

        let v_tile_base = v_bh_base + j_tile * val_dim;
        let v_elems = tile_rows * val_dim;
        let mut idx_v = tid;
        while idx_v < v_elems {
            v_sh[idx_v] = f32::cast_from(v[v_tile_base + idx_v]);
            idx_v += 128;
        }

        sync_cube();

        if valid_i {
            for t in 0..tile_rows {
                let j = j_tile + t;
                let masked = causal && j + seq_q > i + seq_kv;
                if !masked {
                    let k_off = t * head_dim;
                    let v_off = t * val_dim;

                    let mut k_val = Array::<f32>::new(c_d);
                    let mut dot_part = 0.0_f32;
                    #[unroll]
                    for c in 0..c_d {
                        let dd = c * 32 + lane;
                        if dd < head_dim {
                            let kv = k_sh[k_off + dd];
                            k_val[c] = kv;
                            dot_part += q_reg[c] * kv;
                        } else {
                            k_val[c] = 0.0_f32;
                        }
                    }
                    let s_ij = plane_sum(dot_part) * scale;
                    let p_ij = (s_ij - lse_i).exp();

                    let mut dp_part = 0.0_f32;
                    #[unroll]
                    for c in 0..c_v {
                        let dd = c * 32 + lane;
                        if dd < val_dim {
                            dp_part += do_reg[c] * v_sh[v_off + dd];
                        }
                    }
                    let dp = plane_sum(dp_part);
                    let s_ds = scale * p_ij * (dp - d_i);

                    #[unroll]
                    for c in 0..c_d {
                        dq_acc[c] += s_ds * k_val[c];
                    }
                }
            }
        }

        sync_cube();
        j_tile += 32;
    }

    if valid_i {
        #[unroll]
        for c in 0..c_d {
            let dd = c * 32 + lane;
            if dd < head_dim {
                dq[dq_base + dd] = E::cast_from(dq_acc[c]);
            }
        }
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dq_wmma_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dq: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let num_d_tiles = comptime!(head_dim / 16);
    let kv_sh_len = comptime!(16 * head_dim);
    let warp_sh_len = comptime!(4 * 256usize);

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let total_q_rows = lse.len();
    let total_bh = total_q_rows / seq_q;
    let q_tiles_per_bh = seq_q.div_ceil(64);

    let cube_id = CUBE_POS;
    let bh = cube_id / q_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let q_tile_idx = cube_id % q_tiles_per_bh;
    let cta_q_base = q_tile_idx * 64;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;
    let warp_q_base = cta_q_base + w * 16;
    let w_off = w * 256;
    let w_end = w_off + 256;

    let q_bh_base = bh * seq_q * head_dim;
    let k_bh_base = bh * seq_kv * head_dim;
    let row_bh_base = bh * seq_q;

    let mut k_sh = Shared::<[E]>::new_aligned_slice(kv_sh_len, 16usize);
    let mut v_sh = Shared::<[E]>::new_aligned_slice(kv_sh_len, 16usize);
    let mut ds_sh = Shared::<[E]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut s_sh = Shared::<[f32]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut dp_sh = Shared::<[f32]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut lse_sh = Shared::<[f32]>::new_aligned_slice(64usize, 16usize);
    let mut d_sh = Shared::<[f32]>::new_aligned_slice(64usize, 16usize);

    if tid < 64 {
        let q_row = cta_q_base + tid;
        if q_row < seq_q {
            lse_sh[tid] = lse[row_bh_base + q_row];
            d_sh[tid] = d[row_bh_base + q_row];
        } else {
            lse_sh[tid] = 0.0_f32;
            d_sh[tid] = 0.0_f32;
        }
    }

    let mut q_frags = Sequence::<cmma::Matrix<E>>::new();
    let mut do_frags = Sequence::<cmma::Matrix<E>>::new();
    let mut dq_frags = Sequence::<cmma::Matrix<f32>>::new();

    #[unroll]
    for d_blk in 0..num_d_tiles {
        let d_off = d_blk * 16;
        #[unroll]
        for step in 0..8usize {
            let idx = lane + step * 32;
            let r = idx / 16;
            let c = idx % 16;
            let q_idx = warp_q_base + r;
            let d_idx = d_off + c;
            if q_idx < seq_q && d_idx < head_dim {
                ds_sh[w_off + idx] = q[q_bh_base + q_idx * head_dim + d_idx];
            } else {
                ds_sh[w_off + idx] = E::cast_from(0.0_f32);
            }
        }
        sync_plane();
        let q_frag = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &ds_sh[w_off..w_end],
            16u32,
        );
        q_frags.push(q_frag);
        sync_plane();

        #[unroll]
        for step in 0..8usize {
            let idx = lane + step * 32;
            let r = idx / 16;
            let c = idx % 16;
            let q_idx = warp_q_base + r;
            let d_idx = d_off + c;
            if q_idx < seq_q && d_idx < head_dim {
                ds_sh[w_off + idx] = do_[q_bh_base + q_idx * head_dim + d_idx];
            } else {
                ds_sh[w_off + idx] = E::cast_from(0.0_f32);
            }
        }
        sync_plane();
        let do_frag = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &ds_sh[w_off..w_end],
            16u32,
        );
        do_frags.push(do_frag);
        sync_plane();

        let dq_frag = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0_f32,
        );
        dq_frags.push(dq_frag);
    }

    sync_cube();

    let cta_q_end = if seq_q < cta_q_base + 64 {
        seq_q
    } else {
        cta_q_base + 64
    };
    let cta_q_max = cta_q_end - 1;
    let max_kv = if causal {
        if cta_q_max + seq_kv < seq_q {
            0usize
        } else {
            let lim = cta_q_max + seq_kv + 1 - seq_q;
            if seq_kv < lim { seq_kv } else { lim }
        }
    } else {
        seq_kv
    };

    let mut kv_base = 0usize;
    while kv_base < max_kv {
        #[unroll]
        for d_blk in 0..num_d_tiles {
            let blk_off = d_blk * 256;
            let d_off = d_blk * 16;
            #[unroll]
            for s in 0..2usize {
                let elem = tid + s * 128;
                let r = elem / 16;
                let c = elem % 16;
                let kv_idx = kv_base + r;
                let d_idx = d_off + c;
                if kv_idx < seq_kv && d_idx < head_dim {
                    let g_off = k_bh_base + kv_idx * head_dim + d_idx;
                    k_sh[blk_off + elem] = k[g_off];
                    v_sh[blk_off + elem] = v[g_off];
                } else {
                    k_sh[blk_off + elem] = E::cast_from(0.0_f32);
                    v_sh[blk_off + elem] = E::cast_from(0.0_f32);
                }
            }
        }

        sync_cube();

        if warp_q_base < seq_q {
            let warp_q_max = warp_q_base + 15;
            let warp_active = !causal || (kv_base + seq_q <= warp_q_max + seq_kv);
            if warp_active {
                let s_frag = cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0.0_f32,
                );
                let dp_frag = cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0.0_f32,
                );

                #[unroll]
                for d_blk in 0..num_d_tiles {
                    let b_start = d_blk * 256;
                    let b_end = b_start + 256;
                    let k_col = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::ColMajor,
                        &k_sh[b_start..b_end],
                        16u32,
                    );
                    let v_col = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::ColMajor,
                        &v_sh[b_start..b_end],
                        16u32,
                    );
                    cmma::execute(&q_frags[d_blk], &k_col, &s_frag, &s_frag);
                    cmma::execute(&do_frags[d_blk], &v_col, &dp_frag, &dp_frag);
                }

                cmma::store(
                    &mut s_sh[w_off..w_end],
                    &s_frag,
                    16u32,
                    cmma::MatrixLayout::RowMajor,
                );
                cmma::store(
                    &mut dp_sh[w_off..w_end],
                    &dp_frag,
                    16u32,
                    cmma::MatrixLayout::RowMajor,
                );
                sync_plane();

                #[unroll]
                for step in 0..8usize {
                    let idx = lane + step * 32;
                    let r = idx / 16;
                    let c = idx % 16;
                    let i = warp_q_base + r;
                    let j = kv_base + c;
                    let masked = i >= seq_q || j >= seq_kv || (causal && j + seq_q > i + seq_kv);
                    let ds_val = if masked {
                        0.0_f32
                    } else {
                        let lse_i = lse_sh[w * 16 + r];
                        let d_i = d_sh[w * 16 + r];
                        let s_ij = s_sh[w_off + idx] * scale;
                        let p_ij = (s_ij - lse_i).exp();
                        let dp_ij = dp_sh[w_off + idx];
                        scale * p_ij * (dp_ij - d_i)
                    };
                    ds_sh[w_off + idx] = E::cast_from(ds_val);
                }
                sync_plane();

                let ds_frag = cmma::Matrix::<E>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::RowMajor,
                    &ds_sh[w_off..w_end],
                    16u32,
                );
                #[unroll]
                for d_blk in 0..num_d_tiles {
                    let b_start = d_blk * 256;
                    let b_end = b_start + 256;
                    let k_row = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                        &k_sh[b_start..b_end],
                        16u32,
                    );
                    let dq_acc = &dq_frags[d_blk];
                    cmma::execute(&ds_frag, &k_row, dq_acc, dq_acc);
                }
            }
        }

        sync_cube();
        kv_base += 16;
    }

    if warp_q_base < seq_q {
        #[unroll]
        for d_blk in 0..num_d_tiles {
            let d_off = d_blk * 16;
            cmma::store(
                &mut s_sh[w_off..w_end],
                &dq_frags[d_blk],
                16u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_plane();
            #[unroll]
            for step in 0..8usize {
                let idx = lane + step * 32;
                let r = idx / 16;
                let c = idx % 16;
                let q_idx = warp_q_base + r;
                let d_idx = d_off + c;
                if q_idx < seq_q && d_idx < head_dim {
                    dq[q_bh_base + q_idx * head_dim + d_idx] = E::cast_from(s_sh[w_off + idx]);
                }
            }
            sync_plane();
        }
    }
}

/// Compute `dQ` into a pre-allocated tensor (overwrites, not accumulates).
///
/// Inputs (q/k/v/dO/dQ dtype taken from `global_dtypes`, accumulation in fp32):
/// - `q, k, v`: `[B, H, N, d]`.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
/// - `lse`:     `[B, H, N]` fp32: saved from forward.
/// - `d`:       `[B, H, N]` fp32: from the prepass kernel.
///
/// Output:
/// - `dq`:      `[B, H, N, d]`: written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_dq(
    client: &Client,
    q: TensorBinding,
    k: TensorBinding,
    v: TensorBinding,
    do_: TensorBinding,
    lse: TensorBinding,
    d: TensorBinding,
    dq: TensorBinding,
    global_dtypes: &AttentionGlobalTypes,
    config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    let dtype = io_dtype(global_dtypes)?;
    let f32_size = f32::elem_type_native().size();

    let head_dim = q.shape[q.shape.len() - 1];
    let val_dim = v.shape[v.shape.len() - 1];

    let address_type = q
        .required_address_type(dtype.size())
        .max(k.required_address_type(dtype.size()))
        .max(v.required_address_type(dtype.size()))
        .max(do_.required_address_type(dtype.size()))
        .max(lse.required_address_type(f32_size))
        .max(d.required_address_type(f32_size))
        .max(dq.required_address_type(dtype.size()));

    let has_wmma_16 =
        client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&cubecl::ir::features::MmaConfig {
                a_type: dtype,
                b_type: dtype,
                cd_type: f32::elem_type_native(),
                m: 16,
                k: 16,
                n: 16,
            });

    if client.properties().hardware.plane_size_max == 32
        && has_wmma_16
        && head_dim > 0
        && head_dim <= 128
        && head_dim % 16 == 0
        && val_dim == head_dim
    {
        let seq_q = q.shape[q.shape.len() - 2];
        let total_q_rows = lse.shape.iter().product::<usize>();
        let total_bh = total_q_rows / seq_q;
        let q_tiles_per_bh = seq_q.div_ceil(64);
        let num_cubes = total_bh * q_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        flash_attention_backward_dq_wmma_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            do_.into_tensor_arg(),
            lse.into_tensor_arg(),
            d.into_tensor_arg(),
            dq.into_tensor_arg(),
            config.scale,
            head_dim,
            config.causal,
            dtype,
        );
    } else if client.properties().hardware.plane_size_max == 32 && head_dim <= 128 && val_dim <= 128
    {
        let seq_q = q.shape[q.shape.len() - 2];
        let total_q_rows = lse.shape.iter().product::<usize>();
        let total_bh = total_q_rows / seq_q;
        let q_tiles_per_bh = seq_q.div_ceil(4);
        let num_cubes = total_bh * q_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        flash_attention_backward_dq_tiled_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            do_.into_tensor_arg(),
            lse.into_tensor_arg(),
            d.into_tensor_arg(),
            dq.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );
    } else {
        let working_units = lse.shape.iter().product::<usize>();
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

        flash_attention_backward_dq_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            do_.into_tensor_arg(),
            lse.into_tensor_arg(),
            d.into_tensor_arg(),
            dq.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );
    }

    Ok(())
}
