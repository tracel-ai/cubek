//! # Blocked Accelerated Householder QR (BAHT)
//!
//! Standard Householder QR applies one reflector at a time to the whole matrix.
//! Each reflector is cheap to compute but touches every remaining row, so the
//! updates form a long sequence of memory-bound rank-1 operations — a poor fit
//! for a GPU that prefers large, arithmetic-intensive GEMMs.
//!
//! BAHT fixes this by batching `tile` reflectors together before touching the
//! trailing matrix, using the **compact WY representation**:
//!
//! ```text
//! H_1 H_2 … H_tile  =  I + V W^T
//! ```
//!
//! where V = [v₁ | v₂ | … | v_tile] stacks the Householder vectors column-wise
//! and W is a matching matrix built up incrementally alongside V.
//!
//! ## Algorithm (one tile of columns)
//!
//! **1. Panel factorization (serial, GPU kernels)**
//!    For each column j in the current tile:
//!    - Compute the Householder vector `vⱼ` and scalar `βⱼ` that zeros out
//!      the sub-diagonal of column j in R.
//!    - Apply the single reflector to the remaining panel columns
//!      (small, cheap kernel — only `tile` columns wide).
//!    - Append `vⱼ` to V and extend W by one column using the recurrence
//!      `wⱼ = −βⱼ (vⱼ + V[:, 0..j] (V[:, 0..j]ᵀ vⱼ))`.
//!
//! **2. Trailing R update (2 GEMMs)**
//!    Apply the accumulated block reflector to all columns right of the panel:
//!    ```text
//!    S          = W^T · R_trailing          // (tile × trailing)
//!    R_trailing -= V · S                    // (rows  × trailing)
//!    ```
//!
//! **3. Q^T update (2 GEMMs)**
//!    Accumulate the same block reflector into Q^T:
//!    ```text
//!    S    = W^T · Q^T                       // (tile × rows)
//!    Q^T += V · S                           // (rows × rows)
//!    ```
//!
//! Steps 2 and 3 are the expensive O(N²) work per tile; by expressing them as
//! two back-to-back `cubek_matmul` GEMM calls they run at near-peak GPU
//! throughput. The serial panel loop (step 1) is O(tile²·rows), which is small
//! when `tile ≪ N`.
//!
//! ## Layout
//! - **R**: column-major `[rows, cols]` — compatible with LAPACK conventions.
//! - **Q^T**: column-major `[rows, rows]` — stored transposed for efficient
//!   left-multiplication in the GEMM updates.
//! - **V, W**: column-major `[rows, tile]`.
//! - **S buffers**: row-major `[tile, *]` — natural output layout of the GEMMs.

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::strategy::Strategy;
use cubek_std::InputBinding;

// ---------------------------------------------------------------------------
// Basic Kernels
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn householder_kernel<F: Float>(
    r: &[F],
    r_offset: u32,
    dim: u32,
    full_rows: u32,
    v: &mut [F],
    beta: &mut [F],
    beta_offset: u32,
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let tdx_usize = tdx as usize;
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);

    let mut c = tdx;
    while c < full_rows {
        v[c as usize] = zero;
        c += CUBE_DIM_X;
    }
    sync_cube();

    let mut product = Shared::<[F]>::new_slice(shared_size);
    product[tdx_usize] = zero;
    sync_cube();

    let mut local_prd = 0.0f64;
    let mut j = tdx;
    let cube_dim_x = CUBE_DIM_X;
    while j < dim {
        let val = f64::cast_from(r[(r_offset + j + 1) as usize]);
        local_prd = fma(val, val, local_prd); // accumulate in f64
        j += cube_dim_x;
    }
    product[tdx_usize] = F::cast_from(local_prd);
    sync_cube();

    let mut pow2 = 1u32;
    while pow2 < cube_dim_x {
        if tdx_usize.is_multiple_of((pow2 as usize) * 2)
            && (tdx_usize + (pow2 as usize) < (cube_dim_x as usize))
        {
            let val = product[tdx_usize + pow2 as usize];
            product[tdx_usize] += val;
        }
        pow2 *= 2;
        sync_cube();
    }

    let is_zero = product[0] == zero;
    if !is_zero {
        if tdx == 0 {
            let r0 = r[r_offset as usize];
            let sigma = product[0];
            // Use FMA for mu = sqrt(r0² + sigma) to avoid cancellation.
            let mu = F::sqrt(fma(r0, r0, sigma));
            let v0 = if r0 <= zero {
                r0 - mu
            } else {
                -sigma / (r0 + mu)
            };
            let v0sq = v0 * v0;
            beta[beta_offset as usize] = two * v0sq / (sigma + v0sq);
            product[0] = v0;
        }
        sync_cube();

        let b = beta[beta_offset as usize];
        let mut j2 = tdx;
        while j2 < dim {
            if b != zero {
                v[(j2 + 1) as usize] = r[(r_offset + j2 + 1) as usize] / product[0];
            } else {
                v[(j2 + 1) as usize] = r[(r_offset + j2 + 1) as usize];
            }
            j2 += cube_dim_x;
        }
    } else {
        let mut j2 = tdx;
        while j2 < dim {
            v[(j2 + 1) as usize] = r[(r_offset + j2 + 1) as usize];
            j2 += cube_dim_x;
        }
        if tdx == 0 {
            beta[beta_offset as usize] = zero;
        }
    }

    if tdx == 0 {
        v[0] = one;
    }
}

#[cube(launch_unchecked)]
fn left_update_r_kernel<F: Float + CubeElement>(
    rows: u32,
    cols: u32,
    row_offset: u32,
    col_offset: u32,
    r: &mut Tensor<F>,
    v: &[F],
    beta: &[F],
    beta_offset: u32,
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let col_idx = ABSOLUTE_POS_X;
    let n_update = cols - col_offset;
    let in_range = col_idx < n_update;
    let r_base = (col_offset * rows + row_offset) as usize;
    let n_rows = rows - row_offset;

    let mut shv = Shared::<[F]>::new_slice(shared_size);
    let mut clear_idx = tdx as usize;
    let cube_dim_x = CUBE_DIM_X as usize;
    while clear_idx < shared_size {
        shv[clear_idx] = F::from_int(0);
        clear_idx += cube_dim_x;
    }
    sync_cube();

    let mut load_idx = tdx as usize;
    while load_idx < shared_size {
        if (load_idx as u32) < n_rows {
            shv[load_idx] = v[load_idx];
        } else {
            shv[load_idx] = F::from_int(0);
        }
        load_idx += cube_dim_x;
    }
    sync_cube();

    let beta_val = beta[beta_offset as usize];
    // Accumulate in f64 to reduce rounding; write back as F.
    let mut w = 0.0f64;
    if in_range {
        for i in 0u32..n_rows {
            let r_idx = r_base + i as usize + col_idx as usize * rows as usize;
            let v_val = if (i as usize) < shared_size {
                shv[i as usize]
            } else {
                v[i as usize]
            };
            w = fma(f64::cast_from(r[r_idx]), f64::cast_from(v_val), w);
        }
        w *= f64::cast_from(beta_val);

        for i in 0u32..n_rows {
            let r_idx = r_base + i as usize + col_idx as usize * rows as usize;
            let v_val = if (i as usize) < shared_size {
                shv[i as usize]
            } else {
                v[i as usize]
            };
            r[r_idx] -= F::cast_from(f64::cast_from(v_val) * w);
        }
    }
}

#[cube(launch_unchecked)]
fn copy_v_to_buf_column_major_kernel<F: Float + CubeElement>(
    input: &[F],
    output: &mut [F],
    j: u32,
    row_offset: u32,
    n: u32,
    full_rows: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < n as usize {
        // Output is Column-Major: [rows, tile]
        output[(j * full_rows + (row_offset + idx as u32)) as usize] = input[idx];
    }
}

#[cube(launch_unchecked)]
fn compute_next_w_column_major_kernel<F: Float + CubeElement>(
    rows: u32,
    j: u32,
    beta: &[F],
    v_buf: &[F],
    w_buf: &mut [F],
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let i = ABSOLUTE_POS;
    let zero = F::from_int(0);

    let mut dots = Shared::<[F]>::new_slice(shared_size);
    dots[tdx as usize] = zero;
    sync_cube();

    for k in 0..j {
        let mut local_dot = 0.0f64;
        let mut r = tdx;
        while r < rows {
            // V is Column-Major: [rows, tile]
            local_dot = fma(
                f64::cast_from(v_buf[(k * rows + r) as usize]),
                f64::cast_from(v_buf[(j * rows + r) as usize]),
                local_dot,
            );
            r += CUBE_DIM_X;
        }
        dots[tdx as usize] = F::cast_from(local_dot);
        sync_cube();

        let mut pow2 = 1u32;
        while pow2 < CUBE_DIM_X {
            if (tdx as usize).is_multiple_of((pow2 as usize) * 2) && (tdx + pow2 < CUBE_DIM_X) {
                let val = dots[(tdx + pow2) as usize];
                dots[tdx as usize] += val;
            }
            pow2 *= 2;
            sync_cube();
        }

        let dot_k = dots[0];
        if i < rows as usize {
            // W is Column-Major: [rows, tile]
            let w_k_i = w_buf[(k * rows + i as u32) as usize];
            if k == 0 {
                w_buf[(j * rows + i as u32) as usize] =
                    v_buf[(j * rows + i as u32) as usize] + w_k_i * dot_k;
            } else {
                w_buf[(j * rows + i as u32) as usize] += w_k_i * dot_k;
            }
        }
        sync_cube();
    }

    if i < rows as usize {
        let b = beta[j as usize];
        if j == 0 {
            w_buf[(j * rows + i as u32) as usize] = -b * v_buf[(j * rows + i as u32) as usize];
        } else {
            w_buf[(j * rows + i as u32) as usize] *= -b;
        }
    }
}

#[cube(launch_unchecked)]
fn clear_buffer_kernel<F: Float + CubeElement>(buffer: &mut [F], n: u32) {
    if ABSOLUTE_POS < n as usize {
        buffer[ABSOLUTE_POS] = F::from_int(0);
    }
}

#[cube(launch_unchecked)]
fn update_trailing_r_final_kernel<F: Float + CubeElement>(
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

/// Elementwise Q^T += z_buf (both are [rows, rows] Column-Major).
#[cube(launch_unchecked)]
fn update_qt_from_z_kernel<F: Float + CubeElement>(
    rows: u32,
    qt: &mut Tensor<F>,
    z_buf: &[F],
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    if row < rows && col < rows {
        let idx = col as usize * rows as usize + row as usize;
        qt[idx] += z_buf[idx];
    }
}

#[cube(launch_unchecked)]
fn update_qt_final_kernel<F: Float + CubeElement>(
    rows: u32,
    current_tile: u32,
    qt: &mut Tensor<F>,
    v_buf: &[F],
    s_buf: &[F],
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    if row < rows && col < rows {
        let mut update = 0.0f64;
        for k in 0..current_tile {
            // V is Column-Major: [rows, current_tile]. (row, k) -> k * rows + row
            let v_val = f64::cast_from(v_buf[(k * rows + row) as usize]);
            // S is Row-Major: [current_tile, rows]. (k, col) -> k * rows + col
            let s_val = f64::cast_from(s_buf[(k * rows + col) as usize]);
            update = fma(v_val, s_val, update);
        }
        // Qt is Column-Major: [rows, rows]. (row, col) -> col * rows + row
        qt[(col * rows + row) as usize] += F::cast_from(update);
    }
}

#[cube(launch_unchecked)]
fn compute_qt_w_kernel<F: Float + CubeElement>(
    rows: u32,
    current_tile: u32,
    qt: &Tensor<F>,
    w_buf: &[F],
    s_buf: &mut [F],
) {
    let j = ABSOLUTE_POS_X; // row index
    let k = ABSOLUTE_POS_Y; // tile index
    if j < rows && k < current_tile {
        let mut sum = 0.0f64;
        for l in 0..rows {
            // Qt is Column-Major: [rows, rows]. (l, j) -> j * rows + l
            let qt_val = f64::cast_from(qt[(j * rows + l) as usize]);
            // W is Column-Major: [rows, current_tile]. (l, k) -> k * rows + l
            let w_val = f64::cast_from(w_buf[(k * rows + l) as usize]);
            sum = fma(qt_val, w_val, sum);
        }
        // S is Row-Major: [current_tile, rows]. (k, j) -> k * rows + j
        s_buf[(k * rows + j) as usize] = F::cast_from(sum);
    }
}

pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q_handle: &TensorHandle<R>,
    r_handle: &TensorHandle<R>,
) {
    let rows = r_handle.shape()[0] as u32;
    let cols = r_handle.shape()[1] as u32;

    let hardware = &client.properties().hardware;
    let shared_mem_limit = hardware.max_shared_memory_size;
    let thread_block_size = (hardware.max_cube_dim.0 as f64).sqrt() as u32;
    let bytes_per_elem = core::mem::size_of::<E>();

    let max_tile_from_shared = ((shared_mem_limit / bytes_per_elem) as f64).sqrt() as u32;
    let mut tile = 1u32;
    while tile * 2 <= max_tile_from_shared {
        tile *= 2;
    }
    let tile = tile.min(cols);

    let num_tiles = cols.div_ceil(tile);
    let dtype = E::as_type_native_unchecked();
    let storage_dtype = dtype.storage_type();

    let beta         = TensorHandle::<R>::zeros(client, vec![tile as usize], dtype);
    let v_tmp        = TensorHandle::<R>::zeros(client, vec![rows as usize], dtype);
    let v_buf_global = TensorHandle::<R>::zeros(client, vec![rows as usize, tile as usize], dtype);
    let w_buf_global = TensorHandle::<R>::zeros(client, vec![rows as usize, tile as usize], dtype);
    let s_tile_global = TensorHandle::<R>::zeros(client, vec![tile as usize, rows as usize], dtype);
    let s_buf_global = TensorHandle::<R>::zeros(client, vec![tile as usize, cols as usize], dtype);
    // z_buf_global must hold both [rows × trailing_cols] (R update) and [rows × rows] (Q^T update).
    let z_buf_global = TensorHandle::<R>::zeros(client, vec![rows as usize, rows as usize], dtype);

    let max_cube_dim = client.properties().hardware.max_cube_dim.0.min(256);
    let mut matmul_dtypes = MatmulElems::from_single_dtype(dtype);
    let cube_dim_2d = CubeDim::new_2d(thread_block_size, thread_block_size);
    let is_f64 = dtype.size() == 8;
    let strategy = if is_f64 { Strategy::Auto } else { Strategy::SimpleUnit(Default::default()) };

    for k in 0..num_tiles {
        let col_start = k * tile;
        let current_tile = tile.min(cols - col_start);

        // V and W are Column-Major: [rows, current_tile]. Stride [1, rows]
        let v_buf: TensorHandle<R> = TensorHandle::new(
            v_buf_global.handle.clone(),
            vec![rows as usize, current_tile as usize],
            vec![1, rows as usize],
            dtype,
        );
        let w_buf: TensorHandle<R> = TensorHandle::new(
            w_buf_global.handle.clone(),
            vec![rows as usize, current_tile as usize],
            vec![1, rows as usize],
            dtype,
        );
        let n_clear = rows * current_tile;
        let cd_clear = CubeDim::new_1d(max_cube_dim);
        let cc_clear = calculate_cube_count_elemwise(client, n_clear as usize, cd_clear);
        unsafe {
            clear_buffer_kernel::launch_unchecked::<E, R>(
                client,
                cc_clear.clone(),
                cd_clear,
                BufferArg::from_raw_parts(v_buf_global.handle.clone(), n_clear as usize),
                n_clear,
            );
            clear_buffer_kernel::launch_unchecked::<E, R>(
                client,
                cc_clear,
                cd_clear,
                BufferArg::from_raw_parts(w_buf_global.handle.clone(), n_clear as usize),
                n_clear,
            );
        }

        for j in 0..current_tile {
            let col = col_start + j;
            let rows_below = rows - col - 1;
            let r_offset = (col * rows + col) as u64;

            let cd = CubeDim::new_1d(max_cube_dim);
            let cc = calculate_cube_count_elemwise(client, 1usize, cd);
            unsafe {
                householder_kernel::launch_unchecked::<E, R>(
                    client,
                    cc,
                    cd,
                    BufferArg::from_raw_parts(r_handle.handle.clone(), (rows * cols) as usize),
                    r_offset as u32,
                    rows_below,
                    rows - col,
                    BufferArg::from_raw_parts(v_tmp.handle.clone(), rows as usize),
                    BufferArg::from_raw_parts(beta.handle.clone(), tile as usize),
                    j,
                    max_cube_dim as usize,
                );
            }

            let n_upd = (col_start + current_tile - col) as usize;
            if n_upd > 0 {
                let cd_upd = CubeDim::new_1d((n_upd as u32).min(max_cube_dim));
                let cc_upd = calculate_cube_count_elemwise(client, n_upd, cd_upd);
                unsafe {
                    left_update_r_kernel::launch_unchecked::<E, R>(
                        client,
                        cc_upd,
                        cd_upd,
                        rows,
                        col_start + current_tile,
                        col,
                        col,
                        r_handle.clone().into_arg(),
                        BufferArg::from_raw_parts(v_tmp.handle.clone(), (rows - col) as usize),
                        BufferArg::from_raw_parts(beta.handle.clone(), tile as usize),
                        j,
                        max_cube_dim as usize,
                    );
                }
            }

            let cd_copy = CubeDim::new_1d(rows.min(max_cube_dim));
            let cc_copy = calculate_cube_count_elemwise(client, (rows - col) as usize, cd_copy);
            unsafe {
                copy_v_to_buf_column_major_kernel::launch_unchecked::<E, R>(
                    client,
                    cc_copy,
                    cd_copy,
                    BufferArg::from_raw_parts(v_tmp.handle.clone(), rows as usize),
                    BufferArg::from_raw_parts(v_buf.handle.clone(), (rows * current_tile) as usize),
                    j,
                    col,
                    rows - col,
                    rows,
                );
            }

            let cd_w = CubeDim::new_1d(rows.min(max_cube_dim));
            let cc_w = calculate_cube_count_elemwise(client, rows as usize, cd_w);
            unsafe {
                compute_next_w_column_major_kernel::launch_unchecked::<E, R>(
                    client,
                    cc_w,
                    cd_w,
                    rows,
                    j,
                    BufferArg::from_raw_parts(beta.handle.clone(), tile as usize),
                    BufferArg::from_raw_parts(v_buf.handle.clone(), (rows * current_tile) as usize),
                    BufferArg::from_raw_parts(w_buf.handle.clone(), (rows * current_tile) as usize),
                    max_cube_dim as usize,
                );
            }
        }

        // AFTER PANEL: Big GEMM updates for trailing R and Q.
        // R trailing update uses launch_ref (optimized GEMM); Q update keeps original kernels.
        // No client.sync() per tile — kernels queue asynchronously.

        // 1. Update Trailing R: R = R - V * (W^T * R_trailing)
        if col_start + current_tile < cols {
            let trailing_cols = cols - (col_start + current_tile);

            // S = W^T * R_trailing: (tile x rows) * (rows x trailing) -> (tile x trailing)
            // W is Column-Major [rows, tile]. Transpose via swap_dims -> [tile, rows].
            // R_trailing is the sub-block of R: we pass full R but with a shape that exposes
            // only the trailing columns. Because R is column-major [rows, cols], the trailing
            // columns start at flat offset (col_start + current_tile) * rows.
            // We expose this as a [rows, cols] column-major handle — the matmul kernel will
            // use the shape from the TensorBinding, which is [rows, trailing_cols].
            let r_trailing_offset_bytes =
                (col_start + current_tile) as u64 * rows as u64 * bytes_per_elem as u64;
            let r_trailing: TensorHandle<R> = TensorHandle::new(
                r_handle.handle.clone().offset_start(r_trailing_offset_bytes),
                vec![rows as usize, trailing_cols as usize],
                vec![1, rows as usize],
                dtype,
            );
            let s_view: TensorHandle<R> = TensorHandle::new(
                s_buf_global.handle.clone(),
                vec![current_tile as usize, trailing_cols as usize],
                vec![trailing_cols as usize, 1],
                dtype,
            );
            let mut w_t = InputBinding::Normal(w_buf.clone().binding(), storage_dtype);
            w_t.swap_dims(0, 1); // [tile, rows] row-major
            cubek_matmul::launch::launch_ref(
                &strategy,
                client,
                w_t,
                InputBinding::Normal(r_trailing.clone().binding(), storage_dtype),
                s_view.clone().binding(),
                &mut matmul_dtypes,
            )
            .unwrap();

            // Z = V * S: (rows x tile) * (tile x trailing) -> (rows x trailing)
            let z_view: TensorHandle<R> = TensorHandle::new(
                z_buf_global.handle.clone(),
                vec![rows as usize, trailing_cols as usize],
                vec![trailing_cols as usize, 1],
                dtype,
            );
            cubek_matmul::launch::launch_ref(
                &strategy,
                client,
                InputBinding::Normal(v_buf.clone().binding(), storage_dtype),
                InputBinding::Normal(s_view.clone().binding(), storage_dtype),
                z_view.clone().binding(),
                &mut matmul_dtypes,
            )
            .unwrap();

            // R_trailing = R_trailing - Z  (element-wise, indexing into global R)
            let cc_r = CubeCount::new_2d(
                rows.div_ceil(thread_block_size),
                trailing_cols.div_ceil(thread_block_size),
            );
            unsafe {
                update_trailing_r_final_kernel::launch_unchecked::<E, R>(
                    client,
                    cc_r,
                    cube_dim_2d,
                    rows,
                    cols,
                    col_start + current_tile,
                    r_handle.clone().into_arg(),
                    BufferArg::from_raw_parts(
                        z_view.handle.clone(),
                        (rows * trailing_cols) as usize,
                    ),
                );
            }
        }

        // 2. Update Q^T: Q^T = Q^T + V * (W^T * Q^T)
        //
        // Step A: S_tile = W^T * Q^T  →  [tile, rows] row-major
        //   W^T is [tile, rows] (swap_dims of w_buf [rows, tile] col-major)
        //   Q^T is [rows, rows] col-major (stride [1, rows])
        //   S is   [tile, rows] row-major (stride [rows, 1])
        let s_tile_qt: TensorHandle<R> = TensorHandle::new(
            s_tile_global.handle.clone(),
            vec![current_tile as usize, rows as usize],
            vec![rows as usize, 1],
            dtype,
        );
        let mut w_t_qt = InputBinding::Normal(w_buf.clone().binding(), storage_dtype);
        w_t_qt.swap_dims(0, 1); // [tile, rows]
        cubek_matmul::launch::launch_ref(
            &strategy,
            client,
            w_t_qt,
            InputBinding::Normal(q_handle.clone().binding(), storage_dtype),
            s_tile_qt.clone().binding(),
            &mut matmul_dtypes,
        )
        .unwrap();

        // Step B: Z_qt = V * S_tile  →  [rows, rows] col-major
        //   V is [rows, tile] col-major (stride [1, rows])
        //   S is [tile, rows] row-major (stride [rows, 1])
        //   Z is [rows, rows] col-major (stride [1, rows]) — matches Q^T layout
        let z_qt: TensorHandle<R> = TensorHandle::new(
            z_buf_global.handle.clone(),
            vec![rows as usize, rows as usize],
            vec![1, rows as usize],
            dtype,
        );
        cubek_matmul::launch::launch_ref(
            &strategy,
            client,
            InputBinding::Normal(v_buf.clone().binding(), storage_dtype),
            InputBinding::Normal(s_tile_qt.clone().binding(), storage_dtype),
            z_qt.clone().binding(),
            &mut matmul_dtypes,
        )
        .unwrap();

        // Step C: Q^T += Z_qt  (both col-major [rows × rows])
        let cc_add = CubeCount::new_2d(
            rows.div_ceil(thread_block_size),
            rows.div_ceil(thread_block_size),
        );
        unsafe {
            update_qt_from_z_kernel::launch_unchecked::<E, R>(
                client,
                cc_add,
                cube_dim_2d,
                rows,
                q_handle.clone().into_arg(),
                BufferArg::from_raw_parts(z_qt.handle.clone(), (rows * rows) as usize),
            );
        }
    }
}
