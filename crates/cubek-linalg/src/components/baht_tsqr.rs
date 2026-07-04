//! # TSQR-inspired blocked Householder QR
//!
//! Factorizes the whole panel with two kernel dispatches per column — a
//! reflector build and a panel update, both parallel across a full cube with
//! shared-memory tree reductions — builds the block reflector's T matrix from
//! a V^T·V Gram GEMM, and applies the trailing updates through GEMMs.

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::strategy::Strategy;
use cubek_std::InputBinding;

use crate::routines::BahtTsqrLaunchSettings;

#[cube(launch_unchecked)]
fn clear_buffer_kernel<F: Float + CubeElement>(buf: &mut [F], n: u32) {
    let idx = ABSOLUTE_POS_X;
    if idx < n {
        buf[idx as usize] = F::cast_from(0.0);
    }
}

/// Compute the Householder reflector for one panel column and write it
/// directly into column `j` of `v_buf` (col-major `[rows, tile]`, cleared at
/// the start of the tile) with the conventional implicit-1 head:
/// `v_buf[j*rows + col] = 1`, tail `= r[k]/v0`.
///
/// One cube; the `sigma = ||r_tail||²` reduction is tree-reduced in shared
/// memory and the normalization is parallel over the cube.
#[cube(launch_unchecked)]
fn householder_kernel<F: Float + CubeElement>(
    r: &[F],
    r_offset: u32,
    dim: u32,
    rows: u32,
    col: u32,
    v_buf: &mut [F],
    beta_vec: &mut [F],
    j: u32,
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let zero = F::cast_from(0.0);
    let one = F::cast_from(1.0);
    let two = F::cast_from(2.0);

    let mut reduce = Shared::<[F]>::new_slice(shared_size);
    let mut local = zero;
    let mut i = tdx + 1;
    while i < dim {
        let v = r[(r_offset + i) as usize];
        local = fma(v, v, local);
        i += cube_dim_x;
    }
    reduce[tdx as usize] = local;
    sync_cube();

    let mut pow2 = 1u32;
    while pow2 < cube_dim_x {
        if (tdx as usize).is_multiple_of((pow2 as usize) * 2) && (tdx + pow2 < cube_dim_x) {
            let val = reduce[(tdx + pow2) as usize];
            reduce[tdx as usize] += val;
        }
        pow2 *= 2;
        sync_cube();
    }
    let sigma = reduce[0];

    if tdx == 0 {
        if sigma == zero {
            beta_vec[j as usize] = zero;
            reduce[0] = one;
        } else {
            let r0 = r[r_offset as usize];
            let mu = F::sqrt(fma(r0, r0, sigma));
            let v0 = if r0 <= zero { r0 - mu } else { -sigma / (r0 + mu) };
            let v0sq = v0 * v0;
            beta_vec[j as usize] = -two * v0sq / (v0sq + sigma);
            reduce[0] = v0;
        }
    }
    sync_cube();
    let v0 = reduce[0];

    let v_base = (j * rows + col) as usize;
    let mut k = tdx;
    while k < dim {
        if k == 0 {
            v_buf[v_base] = one;
        } else {
            v_buf[v_base + k as usize] = r[(r_offset + k) as usize] / v0;
        }
        k += cube_dim_x;
    }
}

/// Apply the reflector held in `v_buf` column `j` to the remaining panel
/// columns of R. One cube per target column: the `v·r` dot product is
/// tree-reduced in shared memory, then the rank-1 update runs parallel over
/// the cube. The reflector head is the explicit 1 written by
/// [`householder_kernel`], so the dot spans the full `dim` range uniformly.
#[cube(launch_unchecked)]
fn apply_householder_kernel<F: Float + CubeElement>(
    rows: u32,
    col: u32,
    r: &mut Tensor<F>,
    v_buf: &[F],
    v_offset: u32,
    beta_vec: &[F],
    j: u32,
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let target_col = col + CUBE_POS_X;
    let dim = rows - col;
    let r_base = (target_col * rows + col) as usize;

    let mut reduce = Shared::<[F]>::new_slice(shared_size);
    let mut local = F::cast_from(0.0);
    let mut k = tdx;
    while k < dim {
        local = fma(
            v_buf[(v_offset + k) as usize],
            r[r_base + k as usize],
            local,
        );
        k += cube_dim_x;
    }
    reduce[tdx as usize] = local;
    sync_cube();

    let mut pow2 = 1u32;
    while pow2 < cube_dim_x {
        if (tdx as usize).is_multiple_of((pow2 as usize) * 2) && (tdx + pow2 < cube_dim_x) {
            let val = reduce[(tdx + pow2) as usize];
            reduce[tdx as usize] += val;
        }
        pow2 *= 2;
        sync_cube();
    }
    let dot_f = reduce[0] * beta_vec[j as usize];

    let mut k2 = tdx;
    while k2 < dim {
        r[r_base + k2 as usize] += v_buf[(v_offset + k2) as usize] * dot_f;
        k2 += cube_dim_x;
    }
}

/// Build the block reflector's T matrix from the Gram matrix. Columns of T
/// depend on all previous columns, so `j` advances sequentially with a cube
/// sync per step, but the rows `i < j` within a column are independent and
/// are computed in parallel across the cube.
#[cube(launch_unchecked)]
fn build_t_tsqr_kernel<F: Float + CubeElement>(
    tile: u32,
    current_tile: u32,
    gram: &[F],
    beta_vec: &[F],
    t_mat: &mut [F],
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let zero = F::cast_from(0.0);

    let total = tile * tile;
    let mut idx = tdx;
    while idx < total {
        t_mat[idx as usize] = zero;
        idx += cube_dim_x;
    }
    sync_cube();

    for j in 0u32..current_tile {
        if tdx == 0 {
            t_mat[(j * tile + j) as usize] = beta_vec[j as usize];
        }
        let mut i = tdx;
        while i < j {
            let mut sum = zero;
            for k in i..j {
                sum = fma(
                    t_mat[(k * tile + i) as usize],
                    gram[(k * tile + j) as usize],
                    sum,
                );
            }
            t_mat[(j * tile + i) as usize] = beta_vec[j as usize] * sum;
            i += cube_dim_x;
        }
        sync_cube();
    }
}

#[cube(launch_unchecked)]
fn update_trailing_r_kernel<F: Float + CubeElement>(
    rows: u32,
    cols: u32,
    col_start_trailing: u32,
    r: &mut Tensor<F>,
    z_buf: &[F],
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    let trailing_cols = cols - col_start_trailing;
    if row < rows && col < trailing_cols {
        let r_idx = (col_start_trailing + col) as usize * rows as usize + row as usize;
        let z_idx = row as usize * trailing_cols as usize + col as usize;
        r[r_idx] += z_buf[z_idx];
    }
}

#[cube(launch_unchecked)]
fn update_qt_from_z_kernel<F: Float + CubeElement>(rows: u32, qt: &mut Tensor<F>, z_buf: &[F]) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    if row < rows && col < rows {
        let idx = col as usize * rows as usize + row as usize;
        qt[idx] += z_buf[idx];
    }
}

/// Launch QR decomposition using the TSQR-inspired kernels and GEMM updates.
pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q_handle: &TensorHandle<R>,
    r_handle: &TensorHandle<R>,
    settings: BahtTsqrLaunchSettings,
) {
    let rows = r_handle.shape()[0] as u32;
    let cols = r_handle.shape()[1] as u32;

    let BahtTsqrLaunchSettings {
        tile,
        max_cube_dim,
        thread_block_size,
        cube_dim_2d,
        strategy_gram,
        strategy_w,
        strategy_tall,
    } = settings;

    let num_tiles = cols.div_ceil(tile);
    let dtype = E::as_type_native_unchecked();
    let storage_dtype = dtype.storage_type();

    let beta_vec = TensorHandle::<R>::zeros(client, vec![tile as usize], dtype);
    let v_buf_global = TensorHandle::<R>::zeros(client, vec![rows as usize, tile as usize], dtype);
    let w_buf_global = TensorHandle::<R>::zeros(client, vec![rows as usize, tile as usize], dtype);
    let gram_buf_global = TensorHandle::<R>::zeros(client, vec![tile as usize, tile as usize], dtype);
    let t_buf_global = TensorHandle::<R>::zeros(client, vec![tile as usize, tile as usize], dtype);
    let s_buf_global = TensorHandle::<R>::zeros(client, vec![tile as usize, cols as usize], dtype);
    let s_tile_global = TensorHandle::<R>::zeros(client, vec![tile as usize, rows as usize], dtype);
    let z_buf_global = TensorHandle::<R>::zeros(client, vec![rows as usize, rows as usize], dtype);

    let mut matmul_dtypes = MatmulElems::from_single_dtype(dtype);

    let launch_matmul = |strategy: &Strategy, lhs: InputBinding<R>, rhs: InputBinding<R>, out: TensorBinding<R>, dtypes: &mut MatmulElems| {
        cubek_matmul::launch::launch_ref(strategy, client, lhs, rhs, out, dtypes).unwrap();
    };

    for k in 0..num_tiles {
        let col_start = k * tile;
        let current_tile = tile.min(cols - col_start);

        let v_buf = TensorHandle::<R>::new(v_buf_global.handle.clone(), vec![rows as usize, current_tile as usize], vec![1, rows as usize], dtype);
        let w_buf = TensorHandle::<R>::new(w_buf_global.handle.clone(), vec![rows as usize, current_tile as usize], vec![1, rows as usize], dtype);

        let n_clear = rows * current_tile;
        let cd_clear = CubeDim::new_1d(max_cube_dim);
        let cc_clear = calculate_cube_count_elemwise(client, n_clear as usize, cd_clear);
        unsafe {
            clear_buffer_kernel::launch_unchecked::<E, R>(client, cc_clear.clone(), cd_clear, BufferArg::from_raw_parts(v_buf_global.handle.clone(), n_clear as usize), n_clear);
        }

        let cd_panel = CubeDim::new_1d(max_cube_dim);
        let shared_size = max_cube_dim as usize;

        for j in 0..current_tile {
            let col = col_start + j;
            let dim = rows - col;
            let r_offset = col * rows + col;

            unsafe {
                householder_kernel::launch_unchecked::<E, R>(client, CubeCount::new_1d(1), cd_panel, BufferArg::from_raw_parts(r_handle.handle.clone(), (rows * cols) as usize), r_offset, dim, rows, col, BufferArg::from_raw_parts(v_buf.handle.clone(), (rows * current_tile) as usize), BufferArg::from_raw_parts(beta_vec.handle.clone(), tile as usize), j, shared_size);
            }

            // One cube per remaining panel column, cube-wide reduction inside.
            let n_upd = current_tile - j;
            unsafe {
                apply_householder_kernel::launch_unchecked::<E, R>(client, CubeCount::new_1d(n_upd), cd_panel, rows, col, r_handle.clone().into_arg(), BufferArg::from_raw_parts(v_buf.handle.clone(), (rows * current_tile) as usize), j * rows + col, BufferArg::from_raw_parts(beta_vec.handle.clone(), tile as usize), j, shared_size);
            }
        }

        let mut v_t_gram = InputBinding::Normal(v_buf.clone().binding(), storage_dtype);
        v_t_gram.swap_dims(0, 1);
        launch_matmul(&strategy_gram, v_t_gram, InputBinding::Normal(v_buf.clone().binding(), storage_dtype), TensorHandle::<R>::new(gram_buf_global.handle.clone(), vec![current_tile as usize, current_tile as usize], vec![tile as usize, 1], dtype).binding(), &mut matmul_dtypes);

        unsafe {
            build_t_tsqr_kernel::launch_unchecked::<E, R>(client, CubeCount::new_1d(1), CubeDim::new_1d(tile), tile, current_tile, BufferArg::from_raw_parts(gram_buf_global.handle.clone(), (tile * tile) as usize), BufferArg::from_raw_parts(beta_vec.handle.clone(), tile as usize), BufferArg::from_raw_parts(t_buf_global.handle.clone(), (tile * tile) as usize));
        }

        let t_buf = TensorHandle::<R>::new(t_buf_global.handle.clone(), vec![current_tile as usize, current_tile as usize], vec![1, tile as usize], dtype);
        launch_matmul(&strategy_w, InputBinding::Normal(v_buf.clone().binding(), storage_dtype), InputBinding::Normal(t_buf.clone().binding(), storage_dtype), w_buf.clone().binding(), &mut matmul_dtypes);

        let has_trailing = col_start + current_tile < cols;
        let trailing_cols = if has_trailing { cols - (col_start + current_tile) } else { 0 };
        let r_trail_offset = (col_start + current_tile) as u64 * rows as u64 * core::mem::size_of::<E>() as u64;

        if has_trailing {
            let r_trailing = TensorHandle::<R>::new(r_handle.handle.clone().offset_start(r_trail_offset), vec![rows as usize, trailing_cols as usize], vec![1, rows as usize], dtype);
            let s_r = TensorHandle::<R>::new(s_buf_global.handle.clone(), vec![current_tile as usize, trailing_cols as usize], vec![trailing_cols as usize, 1], dtype);
            let z_r = TensorHandle::<R>::new(z_buf_global.handle.clone(), vec![rows as usize, trailing_cols as usize], vec![trailing_cols as usize, 1], dtype);
            let mut w_t = InputBinding::Normal(w_buf.clone().binding(), storage_dtype);
            w_t.swap_dims(0, 1);
            launch_matmul(&strategy_tall, w_t, InputBinding::Normal(r_trailing.clone().binding(), storage_dtype), s_r.clone().binding(), &mut matmul_dtypes);
            launch_matmul(&strategy_tall, InputBinding::Normal(v_buf.clone().binding(), storage_dtype), InputBinding::Normal(s_r.clone().binding(), storage_dtype), z_r.clone().binding(), &mut matmul_dtypes);
            let cc_r = CubeCount::new_2d(rows.div_ceil(thread_block_size), trailing_cols.div_ceil(thread_block_size));
            unsafe {
                update_trailing_r_kernel::launch_unchecked::<E, R>(client, cc_r, cube_dim_2d, rows, cols, col_start + current_tile, r_handle.clone().into_arg(), BufferArg::from_raw_parts(z_r.handle.clone(), (rows * trailing_cols) as usize));
            }
        }

        let s_tile_qt = TensorHandle::<R>::new(s_tile_global.handle.clone(), vec![current_tile as usize, rows as usize], vec![rows as usize, 1], dtype);
        let mut w_t2 = InputBinding::Normal(w_buf.clone().binding(), storage_dtype);
        w_t2.swap_dims(0, 1);
        launch_matmul(&strategy_tall, w_t2, InputBinding::Normal(q_handle.clone().binding(), storage_dtype), s_tile_qt.clone().binding(), &mut matmul_dtypes);
        let z_qt = TensorHandle::<R>::new(z_buf_global.handle.clone(), vec![rows as usize, rows as usize], vec![1, rows as usize], dtype);
        launch_matmul(&strategy_tall, InputBinding::Normal(v_buf.clone().binding(), storage_dtype), InputBinding::Normal(s_tile_qt.clone().binding(), storage_dtype), z_qt.clone().binding(), &mut matmul_dtypes);
        let cc_q = CubeCount::new_2d(rows.div_ceil(thread_block_size), rows.div_ceil(thread_block_size));
        unsafe {
            update_qt_from_z_kernel::launch_unchecked::<E, R>(client, cc_q, cube_dim_2d, rows, q_handle.clone().into_arg(), BufferArg::from_raw_parts(z_qt.handle.clone(), (rows * rows) as usize));
        }
    }
}
