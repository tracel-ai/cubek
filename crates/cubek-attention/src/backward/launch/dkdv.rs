//! Kernel 3: compute `dK` and `dV`. KV-outer loop, accumulates in registers,
//! no atomics.
//!
//! Each program instance owns one KV-block and sweeps all Q-blocks. Both
//! `dK` and `dV` are accumulated independently across the sweep and written
//! once at the end.
//!
//! Implemented as two separate kernels (one for dK, one for dV) because the
//! combined kernel hits the wgpu backend's 8-storage-buffer-per-stage cap.
//! dV doesn't need `V` or `D` so its binding set is naturally smaller; dK
//! drops `dv` instead.

use cubecl::{CubeDim, calculate_cube_count_elemwise, client::Client, prelude::*};

use super::io_dtype;
use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dv_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    dv: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = ABSOLUTE_POS;
    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);

    let total_kv_rows = k.len() / head_dim;
    if row_idx >= total_kv_rows {
        terminate!();
    }

    let j = row_idx % seq_kv;
    let bh = row_idx / seq_kv;

    let k_base = row_idx * head_dim;
    let dv_base = row_idx * val_dim;
    let q_row_base = bh * seq_q * head_dim;
    let do_row_base = bh * seq_q * val_dim;
    let row_idx_base = bh * seq_q;

    let mut dv_acc = Array::<f32>::new(val_dim);
    for dd in 0..val_dim {
        dv_acc[dd] = 0.0_f32;
    }

    for i in 0..seq_q {
        let masked = causal && j + seq_q > i + seq_kv;
        if !masked {
            let q_base = q_row_base + i * head_dim;
            let do_base = do_row_base + i * val_dim;
            let lse_i = lse[row_idx_base + i];

            let mut dot = 0.0_f32;
            for dd in 0..head_dim {
                dot += f32::cast_from(q[q_base + dd]) * f32::cast_from(k[k_base + dd]);
            }
            let p_ij = (dot * scale - lse_i).exp();

            for dd in 0..val_dim {
                dv_acc[dd] += p_ij * f32::cast_from(do_[do_base + dd]);
            }
        }
    }

    for dd in 0..val_dim {
        dv[dv_base + dd] = E::cast_from(dv_acc[dd]);
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dk_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dk: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = ABSOLUTE_POS;
    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);

    let total_kv_rows = k.len() / head_dim;
    if row_idx >= total_kv_rows {
        terminate!();
    }

    let j = row_idx % seq_kv;
    let bh = row_idx / seq_kv;

    let k_base = row_idx * head_dim;
    let v_base = row_idx * val_dim;
    let dk_base = row_idx * head_dim;
    let q_row_base = bh * seq_q * head_dim;
    let do_row_base = bh * seq_q * val_dim;
    let row_idx_base = bh * seq_q;

    let mut dk_acc = Array::<f32>::new(head_dim);
    for dd in 0..head_dim {
        dk_acc[dd] = 0.0_f32;
    }

    for i in 0..seq_q {
        let masked = causal && j + seq_q > i + seq_kv;
        if !masked {
            let q_base = q_row_base + i * head_dim;
            let do_base = do_row_base + i * val_dim;
            let lse_i = lse[row_idx_base + i];
            let d_i = d[row_idx_base + i];

            let mut dot = 0.0_f32;
            for dd in 0..head_dim {
                dot += f32::cast_from(q[q_base + dd]) * f32::cast_from(k[k_base + dd]);
            }
            let p_ij = (dot * scale - lse_i).exp();

            let mut dp = 0.0_f32;
            for dd in 0..val_dim {
                dp += f32::cast_from(do_[do_base + dd]) * f32::cast_from(v[v_base + dd]);
            }

            let s_ds = scale * p_ij * (dp - d_i);
            for dd in 0..head_dim {
                dk_acc[dd] += s_ds * f32::cast_from(q[q_base + dd]);
            }
        }
    }

    for dd in 0..head_dim {
        dk[dk_base + dd] = E::cast_from(dk_acc[dd]);
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dkdv_tiled_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dk: &mut Tensor<E>,
    dv: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let c_d = comptime!(head_dim.div_ceil(32));
    let c_v = comptime!(val_dim.div_ceil(32));
    let q_sh_len = comptime!(32 * head_dim);
    let do_sh_len = comptime!(32 * val_dim);

    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);
    let total_kv_rows = k.len() / head_dim;
    let total_bh = total_kv_rows / seq_kv;
    let kv_tiles_per_bh = seq_kv.div_ceil(4);

    let cube_id = CUBE_POS;
    let bh = cube_id / kv_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let kv_tile_idx = cube_id % kv_tiles_per_bh;
    let j_base = kv_tile_idx * 4;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;

    let j = j_base + w;
    let valid_j = j < seq_kv;
    let kv_row_idx = bh * seq_kv + j;

    let k_base = kv_row_idx * head_dim;
    let v_base = kv_row_idx * val_dim;
    let dk_base = kv_row_idx * head_dim;
    let dv_base = kv_row_idx * val_dim;
    let q_bh_base = bh * seq_q * head_dim;
    let do_bh_base = bh * seq_q * val_dim;
    let row_bh_base = bh * seq_q;

    let mut k_reg = Array::<f32>::new(c_d);
    let mut v_reg = Array::<f32>::new(c_v);
    let mut dk_acc = Array::<f32>::new(c_d);
    let mut dv_acc = Array::<f32>::new(c_v);

    #[unroll]
    for c in 0..c_d {
        let dd = c * 32 + lane;
        if valid_j && dd < head_dim {
            k_reg[c] = f32::cast_from(k[k_base + dd]);
        } else {
            k_reg[c] = 0.0_f32;
        }
        dk_acc[c] = 0.0_f32;
    }

    #[unroll]
    for c in 0..c_v {
        let dd = c * 32 + lane;
        if valid_j && dd < val_dim {
            v_reg[c] = f32::cast_from(v[v_base + dd]);
        } else {
            v_reg[c] = 0.0_f32;
        }
        dv_acc[c] = 0.0_f32;
    }

    let min_q = if causal && j_base + seq_q > seq_kv {
        j_base + seq_q - seq_kv
    } else {
        0usize
    };

    let mut q_sh = Shared::<[f32]>::new_slice(q_sh_len);
    let mut do_sh = Shared::<[f32]>::new_slice(do_sh_len);
    let mut lse_sh = Shared::<[f32]>::new_slice(32usize);
    let mut d_sh = Shared::<[f32]>::new_slice(32usize);

    let mut i_tile = min_q;
    while i_tile < seq_q {
        let rem = seq_q - i_tile;
        let tile_rows = if rem < 32usize { rem } else { 32usize };

        let q_tile_base = q_bh_base + i_tile * head_dim;
        let q_elems = tile_rows * head_dim;
        let mut idx = tid;
        while idx < q_elems {
            q_sh[idx] = f32::cast_from(q[q_tile_base + idx]);
            idx += 128;
        }

        let do_tile_base = do_bh_base + i_tile * val_dim;
        let do_elems = tile_rows * val_dim;
        let mut idx_do = tid;
        while idx_do < do_elems {
            do_sh[idx_do] = f32::cast_from(do_[do_tile_base + idx_do]);
            idx_do += 128;
        }

        if tid < tile_rows {
            lse_sh[tid] = lse[row_bh_base + i_tile + tid];
            d_sh[tid] = d[row_bh_base + i_tile + tid];
        }

        sync_cube();

        if valid_j {
            for t in 0..tile_rows {
                let i = i_tile + t;
                let masked = causal && j + seq_q > i + seq_kv;
                if !masked {
                    let q_off = t * head_dim;
                    let do_off = t * val_dim;
                    let lse_i = lse_sh[t];
                    let d_i = d_sh[t];

                    let mut q_val = Array::<f32>::new(c_d);
                    let mut dot_part = 0.0_f32;
                    #[unroll]
                    for c in 0..c_d {
                        let dd = c * 32 + lane;
                        if dd < head_dim {
                            let qv = q_sh[q_off + dd];
                            q_val[c] = qv;
                            dot_part += qv * k_reg[c];
                        } else {
                            q_val[c] = 0.0_f32;
                        }
                    }
                    let p_ij = (plane_sum(dot_part) * scale - lse_i).exp();

                    let mut dp_part = 0.0_f32;
                    #[unroll]
                    for c in 0..c_v {
                        let dd = c * 32 + lane;
                        if dd < val_dim {
                            let dov = do_sh[do_off + dd];
                            dv_acc[c] += p_ij * dov;
                            dp_part += dov * v_reg[c];
                        }
                    }
                    let dp = plane_sum(dp_part);
                    let s_ds = scale * p_ij * (dp - d_i);

                    #[unroll]
                    for c in 0..c_d {
                        dk_acc[c] += s_ds * q_val[c];
                    }
                }
            }
        }

        sync_cube();
        i_tile += 32;
    }

    if valid_j {
        #[unroll]
        for c in 0..c_d {
            let dd = c * 32 + lane;
            if dd < head_dim {
                dk[dk_base + dd] = E::cast_from(dk_acc[c]);
            }
        }
        #[unroll]
        for c in 0..c_v {
            let dd = c * 32 + lane;
            if dd < val_dim {
                dv[dv_base + dd] = E::cast_from(dv_acc[c]);
            }
        }
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dkdv_wmma_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dk: &mut Tensor<E>,
    dv: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let num_d_tiles = comptime!(head_dim / 16);
    let q_sh_len = comptime!(16 * head_dim);
    let warp_sh_len = comptime!(4 * 256usize);

    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);
    let total_kv_rows = k.len() / head_dim;
    let total_bh = total_kv_rows / seq_kv;
    let kv_tiles_per_bh = seq_kv.div_ceil(64);

    let cube_id = CUBE_POS;
    let bh = cube_id / kv_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let kv_tile_idx = cube_id % kv_tiles_per_bh;
    let cta_kv_base = kv_tile_idx * 64;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;
    let warp_kv_base = cta_kv_base + w * 16;
    let w_off = w * 256;
    let w_end = w_off + 256;

    let q_bh_base = bh * seq_q * head_dim;
    let k_bh_base = bh * seq_kv * head_dim;
    let row_bh_base = bh * seq_q;

    let mut q_sh = Shared::<[E]>::new_aligned_slice(q_sh_len, 16usize);
    let mut do_sh = Shared::<[E]>::new_aligned_slice(q_sh_len, 16usize);
    let mut p_sh = Shared::<[E]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut ds_sh = Shared::<[E]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut s_sh = Shared::<[f32]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut dp_sh = Shared::<[f32]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut lse_sh = Shared::<[f32]>::new_aligned_slice(16usize, 16usize);
    let mut d_sh = Shared::<[f32]>::new_aligned_slice(16usize, 16usize);

    let mut k_col_frags = Sequence::<cmma::Matrix<E>>::new();
    let mut v_col_frags = Sequence::<cmma::Matrix<E>>::new();
    let mut dk_frags = Sequence::<cmma::Matrix<f32>>::new();
    let mut dv_frags = Sequence::<cmma::Matrix<f32>>::new();

    #[unroll]
    for d_blk in 0..num_d_tiles {
        let d_off = d_blk * 16;
        #[unroll]
        for step in 0..8usize {
            let idx = lane + step * 32;
            let r = idx / 16;
            let c = idx % 16;
            let kv_idx = warp_kv_base + r;
            let d_idx = d_off + c;
            if kv_idx < seq_kv && d_idx < head_dim {
                ds_sh[w_off + idx] = k[k_bh_base + kv_idx * head_dim + d_idx];
            } else {
                ds_sh[w_off + idx] = E::cast_from(0.0_f32);
            }
        }
        sync_plane();
        let k_col = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
            &ds_sh[w_off..w_end],
            16u32,
        );
        k_col_frags.push(k_col);
        sync_plane();

        #[unroll]
        for step in 0..8usize {
            let idx = lane + step * 32;
            let r = idx / 16;
            let c = idx % 16;
            let kv_idx = warp_kv_base + r;
            let d_idx = d_off + c;
            if kv_idx < seq_kv && d_idx < head_dim {
                ds_sh[w_off + idx] = v[k_bh_base + kv_idx * head_dim + d_idx];
            } else {
                ds_sh[w_off + idx] = E::cast_from(0.0_f32);
            }
        }
        sync_plane();
        let v_col = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
            &ds_sh[w_off..w_end],
            16u32,
        );
        v_col_frags.push(v_col);
        sync_plane();

        let dk_frag = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0_f32,
        );
        dk_frags.push(dk_frag);

        let dv_frag = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0_f32,
        );
        dv_frags.push(dv_frag);
    }

    sync_cube();

    let min_q = if causal && cta_kv_base + seq_q > seq_kv {
        ((cta_kv_base + seq_q - seq_kv) / 16) * 16
    } else {
        0usize
    };

    let mut q_base = min_q;
    while q_base < seq_q {
        #[unroll]
        for d_blk in 0..num_d_tiles {
            let blk_off = d_blk * 256;
            let d_off = d_blk * 16;
            #[unroll]
            for s in 0..2usize {
                let elem = tid + s * 128;
                let r = elem / 16;
                let c = elem % 16;
                let q_idx = q_base + r;
                let d_idx = d_off + c;
                if q_idx < seq_q && d_idx < head_dim {
                    let g_off = q_bh_base + q_idx * head_dim + d_idx;
                    q_sh[blk_off + elem] = q[g_off];
                    do_sh[blk_off + elem] = do_[g_off];
                } else {
                    q_sh[blk_off + elem] = E::cast_from(0.0_f32);
                    do_sh[blk_off + elem] = E::cast_from(0.0_f32);
                }
            }
        }

        if tid < 16 {
            let q_row = q_base + tid;
            if q_row < seq_q {
                lse_sh[tid] = lse[row_bh_base + q_row];
                d_sh[tid] = d[row_bh_base + q_row];
            } else {
                lse_sh[tid] = 0.0_f32;
                d_sh[tid] = 0.0_f32;
            }
        }

        sync_cube();

        if warp_kv_base < seq_kv {
            let q_max = q_base + 15;
            let warp_active = !causal || (warp_kv_base + seq_q <= q_max + seq_kv);
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
                    let q_a = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                        &q_sh[b_start..b_end],
                        16u32,
                    );
                    let do_a = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                        &do_sh[b_start..b_end],
                        16u32,
                    );
                    cmma::execute(&q_a, &k_col_frags[d_blk], &s_frag, &s_frag);
                    cmma::execute(&do_a, &v_col_frags[d_blk], &dp_frag, &dp_frag);
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
                    let i = q_base + r;
                    let j = warp_kv_base + c;
                    let masked = i >= seq_q || j >= seq_kv || (causal && j + seq_q > i + seq_kv);
                    if masked {
                        p_sh[w_off + idx] = E::cast_from(0.0_f32);
                        ds_sh[w_off + idx] = E::cast_from(0.0_f32);
                    } else {
                        let lse_i = lse_sh[r];
                        let d_i = d_sh[r];
                        let s_ij = s_sh[w_off + idx] * scale;
                        let p_ij = (s_ij - lse_i).exp();
                        let dp_ij = dp_sh[w_off + idx];
                        let ds_ij = scale * p_ij * (dp_ij - d_i);
                        p_sh[w_off + idx] = E::cast_from(p_ij);
                        ds_sh[w_off + idx] = E::cast_from(ds_ij);
                    }
                }
                sync_plane();

                let p_t_frag = cmma::Matrix::<E>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::ColMajor,
                    &p_sh[w_off..w_end],
                    16u32,
                );
                let ds_t_frag = cmma::Matrix::<E>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::ColMajor,
                    &ds_sh[w_off..w_end],
                    16u32,
                );

                #[unroll]
                for d_blk in 0..num_d_tiles {
                    let b_start = d_blk * 256;
                    let b_end = b_start + 256;
                    let q_b = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                        &q_sh[b_start..b_end],
                        16u32,
                    );
                    let do_b = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                        &do_sh[b_start..b_end],
                        16u32,
                    );
                    let dv_acc = &dv_frags[d_blk];
                    let dk_acc = &dk_frags[d_blk];
                    cmma::execute(&p_t_frag, &do_b, dv_acc, dv_acc);
                    cmma::execute(&ds_t_frag, &q_b, dk_acc, dk_acc);
                }
            }
        }

        sync_cube();
        q_base += 16;
    }

    if warp_kv_base < seq_kv {
        #[unroll]
        for d_blk in 0..num_d_tiles {
            let d_off = d_blk * 16;
            cmma::store(
                &mut s_sh[w_off..w_end],
                &dk_frags[d_blk],
                16u32,
                cmma::MatrixLayout::RowMajor,
            );
            cmma::store(
                &mut dp_sh[w_off..w_end],
                &dv_frags[d_blk],
                16u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_plane();
            #[unroll]
            for step in 0..8usize {
                let idx = lane + step * 32;
                let r = idx / 16;
                let c = idx % 16;
                let kv_idx = warp_kv_base + r;
                let d_idx = d_off + c;
                if kv_idx < seq_kv && d_idx < head_dim {
                    let g_off = k_bh_base + kv_idx * head_dim + d_idx;
                    dk[g_off] = E::cast_from(s_sh[w_off + idx]);
                    dv[g_off] = E::cast_from(dp_sh[w_off + idx]);
                }
            }
            sync_plane();
        }
    }
}

/// Compute `dK` and `dV` into pre-allocated tensors (overwrites, not
/// accumulates).
///
/// Inputs (q/k/v/dO/dK/dV dtype taken from `global_dtypes`, accumulation in fp32):
/// - `q, k, v`: `[B, H, N, d]`.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
/// - `lse`:     `[B, H, N]` fp32: saved from forward.
/// - `d`:       `[B, H, N]` fp32: from the prepass kernel.
///
/// Outputs:
/// - `dk`:      `[B, H, N, d]`: written cleanly.
/// - `dv`:      `[B, H, N, d]`: written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_dkdv(
    client: &Client,
    q: TensorBinding,
    k: TensorBinding,
    v: TensorBinding,
    do_: TensorBinding,
    lse: TensorBinding,
    d: TensorBinding,
    dk: TensorBinding,
    dv: TensorBinding,
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
        .max(dk.required_address_type(dtype.size()))
        .max(dv.required_address_type(dtype.size()));

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
        let seq_kv = k.shape[k.shape.len() - 2];
        let total_kv_rows = k.shape.iter().product::<usize>() / head_dim;
        let total_bh = total_kv_rows / seq_kv;
        let kv_tiles_per_bh = seq_kv.div_ceil(64);
        let num_cubes = total_bh * kv_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        flash_attention_backward_dkdv_wmma_kernel::launch(
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
            dk.into_tensor_arg(),
            dv.into_tensor_arg(),
            config.scale,
            head_dim,
            config.causal,
            dtype,
        );
    } else if client.properties().hardware.plane_size_max == 32 && head_dim <= 128 && val_dim <= 128
    {
        let seq_kv = k.shape[k.shape.len() - 2];
        let total_kv_rows = k.shape.iter().product::<usize>() / head_dim;
        let total_bh = total_kv_rows / seq_kv;
        let kv_tiles_per_bh = seq_kv.div_ceil(4);
        let num_cubes = total_bh * kv_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        flash_attention_backward_dkdv_tiled_kernel::launch(
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
            dk.into_tensor_arg(),
            dv.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );
    } else {
        let working_units = k.shape.iter().product::<usize>() / head_dim;
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

        flash_attention_backward_dv_kernel::launch(
            client,
            cube_count.clone(),
            cube_dim,
            address_type,
            q.clone().into_tensor_arg(),
            k.clone().into_tensor_arg(),
            do_.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            dv.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );

        flash_attention_backward_dk_kernel::launch(
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
            dk.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );
    }

    Ok(())
}
