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

use crate::definition::QRSetupError;
use crate::routines::{BahtTsqrBlueprint, BahtTsqrLaunchSettings};

#[cube(launch_unchecked)]
fn clear_buffer_kernel<F: Float + CubeElement>(buf: &mut [F], n: u32) {
    let idx = ABSOLUTE_POS_X;
    if idx < n {
        buf[idx as usize] = F::cast_from(0.0);
    }
}

/// Sum-reduce each unit's `local` partial across the cube; every unit
/// returns the total and the shared buffer is free for reuse on return.
///
/// With `use_plane_reduce` (blueprint-gated on plane-op support and a uniform
/// plane size) the partials are folded by the hardware `plane_sum`, one value
/// per plane is merged through `reduce`, and only two barriers are needed.
/// Otherwise a power-of-two shared-memory tree (`log2(cube_dim)` barriers)
/// does the same job on any hardware.
#[cube]
fn cube_reduce_sum<F: Float>(
    reduce: &mut Shared<[F]>,
    local: F,
    tdx: u32,
    cube_dim_x: u32,
    #[comptime] use_plane_reduce: bool,
) -> F {
    let total = if comptime![use_plane_reduce] {
        let plane_total = plane_sum(local);
        if UNIT_POS_PLANE == 0 {
            reduce[(tdx / PLANE_DIM) as usize] = plane_total;
        }
        sync_cube();
        let num_planes = cube_dim_x.div_ceil(PLANE_DIM);
        let mut acc = F::cast_from(0.0);
        for p in 0..num_planes {
            acc += reduce[p as usize];
        }
        acc
    } else {
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
        reduce[0]
    };
    // Every unit finishes reading before the caller may reuse the buffer.
    sync_cube();
    total
}

/// Compute the Householder reflector for one panel column and write it
/// directly into column `j` of `v_buf` (col-major `[rows, tile]`, cleared at
/// the start of the tile) with the conventional implicit-1 head:
/// `v_buf[j*rows + col] = 1`, tail `= r[k]/v0`.
///
/// One cube; the `sigma = ||r_tail||²` reduction runs through
/// [`cube_reduce_sum`] and the normalization is parallel over the cube. All flat offsets
/// are computed in `usize` so `col*rows` cannot wrap on huge matrices.
#[cube(launch_unchecked)]
fn householder_kernel<F: Float + CubeElement>(
    r: &[F],
    rows: u32,
    col: u32,
    v_buf: &mut [F],
    beta_vec: &mut [F],
    j: u32,
    #[comptime] shared_size: usize,
    #[comptime] use_plane_reduce: bool,
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let zero = F::cast_from(0.0);
    let one = F::cast_from(1.0);
    let two = F::cast_from(2.0);

    let dim = rows - col;
    let r_offset = col as usize * rows as usize + col as usize;

    let mut reduce = Shared::<[F]>::new_slice(shared_size);
    let mut local = zero;
    let mut i = tdx + 1;
    while i < dim {
        let v = r[r_offset + i as usize];
        local = fma(v, v, local);
        i += cube_dim_x;
    }

    let sigma = cube_reduce_sum::<F>(&mut reduce, local, tdx, cube_dim_x, use_plane_reduce);

    if tdx == 0 {
        if sigma == zero {
            beta_vec[j as usize] = zero;
            reduce[0] = one;
        } else {
            let r0 = r[r_offset];
            let mu = F::sqrt(fma(r0, r0, sigma));
            let v0 = if r0 <= zero {
                r0 - mu
            } else {
                -sigma / (r0 + mu)
            };
            let v0sq = v0 * v0;
            beta_vec[j as usize] = -two * v0sq / (v0sq + sigma);
            reduce[0] = v0;
        }
    }
    sync_cube();
    let v0 = reduce[0];

    let v_base = j as usize * rows as usize + col as usize;
    let mut k = tdx;
    while k < dim {
        if k == 0 {
            v_buf[v_base] = one;
        } else {
            v_buf[v_base + k as usize] = r[r_offset + k as usize] / v0;
        }
        k += cube_dim_x;
    }
}

/// Apply the reflector held in `v_buf` column `j` to the remaining panel
/// columns of R. One cube per target column: the `v·r` dot product is
/// reduced through [`cube_reduce_sum`], then the rank-1 update runs parallel
/// over the cube. The reflector head is the explicit 1 written by
/// [`householder_kernel`], so the dot spans the full `dim` range uniformly.
#[cube(launch_unchecked)]
fn apply_householder_kernel<F: Float + CubeElement>(
    rows: u32,
    col: u32,
    r: &mut Tensor<F>,
    v_buf: &[F],
    beta_vec: &[F],
    j: u32,
    #[comptime] shared_size: usize,
    #[comptime] use_plane_reduce: bool,
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let target_col = col + CUBE_POS_X;
    let dim = rows - col;
    let r_base = target_col as usize * rows as usize + col as usize;
    let v_base = j as usize * rows as usize + col as usize;

    let mut reduce = Shared::<[F]>::new_slice(shared_size);
    let mut local = F::cast_from(0.0);
    let mut k = tdx;
    while k < dim {
        local = fma(v_buf[v_base + k as usize], r[r_base + k as usize], local);
        k += cube_dim_x;
    }

    let dot_f = cube_reduce_sum::<F>(&mut reduce, local, tdx, cube_dim_x, use_plane_reduce)
        * beta_vec[j as usize];

    let mut k2 = tdx;
    while k2 < dim {
        r[r_base + k2 as usize] += v_buf[v_base + k2 as usize] * dot_f;
        k2 += cube_dim_x;
    }
}

/// Build the block reflector's T matrix from the Gram matrix. Columns of T
/// depend on all previous columns, so `j` advances sequentially with a cube
/// sync per step, but the rows `i < j` within a column are independent and
/// are computed in parallel across the cube.
///
/// T and the Gram matrix are both staged in shared memory: each of the `tile`
/// serialized steps reads back columns written by earlier steps, so building
/// T directly in global memory puts a global round trip inside every step of
/// the dependency chain. The whole working set is `tile * tile` elements
/// (4 KiB at f32/tile=32), so it fits comfortably.
#[cube(launch_unchecked)]
fn build_t_tsqr_kernel<F: Float + CubeElement>(
    current_tile: u32,
    gram: &[F],
    beta_vec: &[F],
    t_mat: &mut [F],
    #[comptime] tile: u32,
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let zero = F::cast_from(0.0);
    let total = comptime!(tile * tile);

    let mut t_sh = Shared::<[F]>::new_slice(comptime!((tile * tile) as usize));
    let mut g_sh = Shared::<[F]>::new_slice(comptime!((tile * tile) as usize));

    let mut idx = tdx;
    while idx < total {
        t_sh[idx as usize] = zero;
        g_sh[idx as usize] = gram[idx as usize];
        idx += cube_dim_x;
    }
    sync_cube();

    for j in 0u32..current_tile {
        if tdx == 0 {
            t_sh[(j * tile + j) as usize] = beta_vec[j as usize];
        }
        let mut i = tdx;
        while i < j {
            let mut sum = zero;
            for k in i..j {
                sum = fma(
                    t_sh[(k * tile + i) as usize],
                    g_sh[(k * tile + j) as usize],
                    sum,
                );
            }
            t_sh[(j * tile + i) as usize] = beta_vec[j as usize] * sum;
            i += cube_dim_x;
        }
        sync_cube();
    }

    let mut o = tdx;
    while o < total {
        t_mat[o as usize] = t_sh[o as usize];
        o += cube_dim_x;
    }
}

/// R's trailing block is col-major while Z is row-major, so this update is a
/// transposing add and cannot be vectorized on both operands at once. Making Z
/// col-major to fix that measurably de-vectorizes the matmul that writes it
/// (its output write drops from `acc_size_4` to `acc_size_1`), which costs
/// more than this kernel saves — so it stays scalar and 2D.
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

/// `dst += z` over a tight contiguous region, launched 1D over `Vector` lanes
/// so the loads/stores are widened. Used for the Q^T update, whose two
/// operands are tight col-major `[rows, rows]` buffers with identical flat
/// indexing.
#[cube(launch_unchecked)]
fn add_assign_kernel<F: Float + CubeElement, N: Size>(
    n_vecs: u32,
    dst: &mut [Vector<F, N>],
    z_buf: &[Vector<F, N>],
) {
    let idx = ABSOLUTE_POS_X;
    if idx < n_vecs {
        dst[idx as usize] += z_buf[idx as usize];
    }
}

/// Launch QR decomposition using the TSQR-inspired kernels and GEMM updates.
pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q_handle: &TensorHandle<R>,
    r_handle: &TensorHandle<R>,
    blueprint: BahtTsqrBlueprint,
    settings: BahtTsqrLaunchSettings,
) -> Result<(), QRSetupError> {
    let rows = r_handle.shape()[0] as u32;
    let cols = r_handle.shape()[1] as u32;
    let use_plane_reduce = blueprint.use_plane_reduce;

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
    let elem_size = storage_dtype.size();

    let rows_us = rows as usize;
    let cols_us = cols as usize;
    let tile_us = tile as usize;

    // Raw scratch allocations: every consumer below re-declares them with
    // hand-computed tight strides (or indexes them flat), and each region is
    // fully written before it is read, so no zero-fill dispatches are needed
    // and no allocator stride policy can interfere.
    let beta_vec = client.empty(tile_us * elem_size);
    let v_buf_global = client.empty(rows_us * tile_us * elem_size);
    let w_buf_global = client.empty(rows_us * tile_us * elem_size);
    let gram_buf_global = client.empty(tile_us * tile_us * elem_size);
    let t_buf_global = client.empty(tile_us * tile_us * elem_size);
    let s_buf_global = client.empty(tile_us * cols_us * elem_size);
    let s_tile_global = client.empty(tile_us * rows_us * elem_size);
    let z_buf_global = client.empty(rows_us * rows_us * elem_size);

    let mut matmul_dtypes = MatmulElems::from_single_dtype(dtype);

    // Widest supported vector size dividing a contiguous element count, plus
    // the resulting lane count. Both operands of every `add_assign_kernel`
    // launch are tight, so divisibility of the count is the only constraint.
    let vec_split = |n: usize| -> (VectorSize, usize) {
        let v = client
            .io_optimized_vector_sizes(elem_size)
            .filter(|v| n.is_multiple_of(*v))
            .max()
            .unwrap_or(1);
        (v, n / v)
    };

    let launch_matmul = |strategy: &Strategy,
                         lhs: InputBinding<R>,
                         rhs: InputBinding<R>,
                         out: TensorBinding<R>,
                         dtypes: &mut MatmulElems|
     -> Result<(), QRSetupError> {
        cubek_matmul::launch::launch_ref(strategy, client, lhs, rhs, out, dtypes)
            .map_err(|e| QRSetupError::Matmul(e.to_string()))
    };

    for k in 0..num_tiles {
        let col_start = k * tile;
        let current_tile = tile.min(cols - col_start);
        let current_tile_us = current_tile as usize;

        let v_buf = TensorHandle::<R>::new(
            v_buf_global.clone(),
            vec![rows_us, current_tile_us],
            vec![1, rows_us],
            dtype,
        );
        let w_buf = TensorHandle::<R>::new(
            w_buf_global.clone(),
            vec![rows_us, current_tile_us],
            vec![1, rows_us],
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
                BufferArg::from_raw_parts(v_buf_global.clone(), n_clear as usize),
                n_clear,
            );
        }

        let cd_panel = CubeDim::new_1d(max_cube_dim);
        let shared_size = max_cube_dim as usize;

        for j in 0..current_tile {
            let col = col_start + j;

            unsafe {
                householder_kernel::launch_unchecked::<E, R>(
                    client,
                    CubeCount::new_1d(1),
                    cd_panel,
                    BufferArg::from_raw_parts(r_handle.handle.clone(), rows_us * cols_us),
                    rows,
                    col,
                    BufferArg::from_raw_parts(v_buf.handle.clone(), rows_us * current_tile_us),
                    BufferArg::from_raw_parts(beta_vec.clone(), tile_us),
                    j,
                    shared_size,
                    use_plane_reduce,
                );
            }

            // One cube per remaining panel column, cube-wide reduction inside.
            let n_upd = current_tile - j;
            unsafe {
                apply_householder_kernel::launch_unchecked::<E, R>(
                    client,
                    CubeCount::new_1d(n_upd),
                    cd_panel,
                    rows,
                    col,
                    r_handle.clone().into_arg(),
                    BufferArg::from_raw_parts(v_buf.handle.clone(), rows_us * current_tile_us),
                    BufferArg::from_raw_parts(beta_vec.clone(), tile_us),
                    j,
                    shared_size,
                    use_plane_reduce,
                );
            }
        }

        let mut v_t_gram = InputBinding::Normal(v_buf.clone().binding(), storage_dtype);
        v_t_gram.swap_dims(0, 1);
        launch_matmul(
            &strategy_gram,
            v_t_gram,
            InputBinding::Normal(v_buf.clone().binding(), storage_dtype),
            TensorHandle::<R>::new(
                gram_buf_global.clone(),
                vec![current_tile_us, current_tile_us],
                vec![tile_us, 1],
                dtype,
            )
            .binding(),
            &mut matmul_dtypes,
        )?;

        unsafe {
            build_t_tsqr_kernel::launch_unchecked::<E, R>(
                client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(tile),
                current_tile,
                BufferArg::from_raw_parts(gram_buf_global.clone(), tile_us * tile_us),
                BufferArg::from_raw_parts(beta_vec.clone(), tile_us),
                BufferArg::from_raw_parts(t_buf_global.clone(), tile_us * tile_us),
                tile,
            );
        }

        let t_buf = TensorHandle::<R>::new(
            t_buf_global.clone(),
            vec![current_tile_us, current_tile_us],
            vec![1, tile_us],
            dtype,
        );
        launch_matmul(
            &strategy_w,
            InputBinding::Normal(v_buf.clone().binding(), storage_dtype),
            InputBinding::Normal(t_buf.clone().binding(), storage_dtype),
            w_buf.clone().binding(),
            &mut matmul_dtypes,
        )?;

        let has_trailing = col_start + current_tile < cols;
        let trailing_cols = if has_trailing {
            cols - (col_start + current_tile)
        } else {
            0
        };
        let r_trail_offset =
            (col_start + current_tile) as u64 * rows as u64 * core::mem::size_of::<E>() as u64;

        if has_trailing {
            let trailing_us = trailing_cols as usize;
            let r_trailing = TensorHandle::<R>::new(
                r_handle.handle.clone().offset_start(r_trail_offset),
                vec![rows_us, trailing_us],
                vec![1, rows_us],
                dtype,
            );
            let s_r = TensorHandle::<R>::new(
                s_buf_global.clone(),
                vec![current_tile_us, trailing_us],
                vec![trailing_us, 1],
                dtype,
            );
            let z_r = TensorHandle::<R>::new(
                z_buf_global.clone(),
                vec![rows_us, trailing_us],
                vec![trailing_us, 1],
                dtype,
            );
            let mut w_t = InputBinding::Normal(w_buf.clone().binding(), storage_dtype);
            w_t.swap_dims(0, 1);
            launch_matmul(
                &strategy_tall,
                w_t,
                InputBinding::Normal(r_trailing.clone().binding(), storage_dtype),
                s_r.clone().binding(),
                &mut matmul_dtypes,
            )?;
            launch_matmul(
                &strategy_tall,
                InputBinding::Normal(v_buf.clone().binding(), storage_dtype),
                InputBinding::Normal(s_r.clone().binding(), storage_dtype),
                z_r.clone().binding(),
                &mut matmul_dtypes,
            )?;
            let cc_r = CubeCount::new_2d(
                rows.div_ceil(thread_block_size),
                trailing_cols.div_ceil(thread_block_size),
            );
            unsafe {
                update_trailing_r_kernel::launch_unchecked::<E, R>(
                    client,
                    cc_r,
                    cube_dim_2d,
                    rows,
                    cols,
                    col_start + current_tile,
                    r_handle.clone().into_arg(),
                    BufferArg::from_raw_parts(z_buf_global.clone(), rows_us * trailing_us),
                );
            }
        }

        let s_tile_qt = TensorHandle::<R>::new(
            s_tile_global.clone(),
            vec![current_tile_us, rows_us],
            vec![rows_us, 1],
            dtype,
        );
        let mut w_t2 = InputBinding::Normal(w_buf.clone().binding(), storage_dtype);
        w_t2.swap_dims(0, 1);
        launch_matmul(
            &strategy_tall,
            w_t2,
            InputBinding::Normal(q_handle.clone().binding(), storage_dtype),
            s_tile_qt.clone().binding(),
            &mut matmul_dtypes,
        )?;
        let z_qt = TensorHandle::<R>::new(
            z_buf_global.clone(),
            vec![rows_us, rows_us],
            vec![1, rows_us],
            dtype,
        );
        launch_matmul(
            &strategy_tall,
            InputBinding::Normal(v_buf.clone().binding(), storage_dtype),
            InputBinding::Normal(s_tile_qt.clone().binding(), storage_dtype),
            z_qt.clone().binding(),
            &mut matmul_dtypes,
        )?;
        let n_q = rows_us * rows_us;
        let (q_vec, n_q_vecs) = vec_split(n_q);
        let cd_q = CubeDim::new_1d(max_cube_dim);
        let cc_q = calculate_cube_count_elemwise(client, n_q_vecs, cd_q);
        unsafe {
            add_assign_kernel::launch_unchecked::<E, R>(
                client,
                cc_q,
                cd_q,
                q_vec,
                n_q_vecs as u32,
                BufferArg::from_raw_parts(q_handle.handle.clone(), n_q_vecs),
                BufferArg::from_raw_parts(z_buf_global.clone(), n_q_vecs),
            );
        }
    }

    Ok(())
}
