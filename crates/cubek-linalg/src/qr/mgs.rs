//! # Modified Gram-Schmidt QR (MGS)
//!
//! Classical Householder/Givens strategies dispatch several kernels (and, for
//! BAHT/TSQR, GEMMs through `cubek_matmul`) per panel — overhead that pays off
//! only once the matrix is large enough to keep a GPU busy. For small
//! matrices that overhead dominates the actual work.
//!
//! MGS instead does the whole factorization with a single persistent cube:
//! one kernel launch per column, looping internally over the previously
//! computed columns. Each column keeps its running vector in shared memory,
//! using the numerically stable *modified* (sequential) variant — project
//! out one already-orthonormal vector at a time rather than projecting the
//! original column against all of them at once (classical Gram-Schmidt).
//!
//! ## Algorithm (column j)
//! ```text
//! v = A[:, j]
//! for i in 0..j:
//!     r[i, j] = dot(q_i, v)
//!     v -= r[i, j] * q_i
//! r[j, j] = ||v||
//! q_j = v / r[j, j]
//! ```
//!
//! ## Layout
//! - **R**: column-major `[rows, cols]`, like the other strategies.
//! - **Q^T** (the `q` buffer): column-major `[rows, rows]`. Only the first
//!   `cols` rows of Q^T (columns of Q) are ever written — they are exactly
//!   the orthonormal vectors produced above. The remaining `rows - cols`
//!   rows of Q^T are left as the identity rows `initialize` seeded them
//!   with. This is sound because every consumer (`Q^T·b` followed by
//!   back-substitution, and the `Q·R = A` reconstruction tests) only ever
//!   combines them with the corresponding rows of R, which this kernel
//!   explicitly zeroes for every column — so those untouched rows are
//!   always multiplied by zero.
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

#[cube(launch_unchecked)]
fn mgs_column_kernel<F: Float + CubeElement>(
    rows: u32,
    j: u32,
    r: &mut Tensor<F>,
    qt: &mut Tensor<F>,
    #[comptime] v_size: usize,
    #[comptime] reduce_size: usize,
) {
    let tdx = UNIT_POS_X;
    let cube_dim_x = CUBE_DIM_X;
    let zero = F::from_int(0);

    let mut v = Shared::<[F]>::new_slice(v_size);
    let mut k = tdx;
    while k < rows {
        v[k as usize] = r[(j * rows + k) as usize];
        k += cube_dim_x;
    }
    sync_cube();

    let mut reduce = Shared::<[F]>::new_slice(reduce_size);

    for i in 0..j {
        let mut local = 0.0f64;
        let mut k2 = tdx;
        while k2 < rows {
            local = fma(
                f64::cast_from(qt[(k2 * rows + i) as usize]),
                f64::cast_from(v[k2 as usize]),
                local,
            );
            k2 += cube_dim_x;
        }
        reduce[tdx as usize] = F::cast_from(local);
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

        let r_ij = reduce[0];
        if tdx == 0 {
            r[(j * rows + i) as usize] = r_ij;
        }

        let mut k3 = tdx;
        while k3 < rows {
            v[k3 as usize] -= r_ij * qt[(k3 * rows + i) as usize];
            k3 += cube_dim_x;
        }
        sync_cube();
    }

    let mut local_norm = 0.0f64;
    let mut k4 = tdx;
    while k4 < rows {
        let val = f64::cast_from(v[k4 as usize]);
        local_norm = fma(val, val, local_norm);
        k4 += cube_dim_x;
    }
    reduce[tdx as usize] = F::cast_from(local_norm);
    sync_cube();

    let mut pow2b = 1u32;
    while pow2b < cube_dim_x {
        if (tdx as usize).is_multiple_of((pow2b as usize) * 2) && (tdx + pow2b < cube_dim_x) {
            let val = reduce[(tdx + pow2b) as usize];
            reduce[tdx as usize] += val;
        }
        pow2b *= 2;
        sync_cube();
    }

    let r_jj = F::sqrt(reduce[0]);
    if tdx == 0 {
        r[(j * rows + j) as usize] = r_jj;
    }
    sync_cube();

    let mut k5 = tdx;
    while k5 < rows {
        if k5 > j {
            r[(j * rows + k5) as usize] = zero;
        }
        k5 += cube_dim_x;
    }

    let inv = if r_jj != zero { F::from_int(1) / r_jj } else { zero };
    let mut k6 = tdx;
    while k6 < rows {
        qt[(k6 * rows + j) as usize] = v[k6 as usize] * inv;
        k6 += cube_dim_x;
    }
}

/// Launch QR decomposition using the Modified Gram-Schmidt kernel.
pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q_handle: &TensorHandle<R>,
    r_handle: &TensorHandle<R>,
) {
    let rows = r_handle.shape()[0] as u32;
    let cols = r_handle.shape()[1] as u32;

    let max_cube_dim = client.properties().hardware.max_cube_dim.0;
    let cube_dim_x = rows.min(max_cube_dim).max(1);
    let cd = CubeDim::new_1d(cube_dim_x);
    let cc = CubeCount::new_1d(1);

    for j in 0..cols {
        unsafe {
            mgs_column_kernel::launch_unchecked::<E, R>(
                client,
                cc.clone(),
                cd,
                rows,
                j,
                r_handle.clone().into_arg(),
                q_handle.clone().into_arg(),
                rows as usize,
                cube_dim_x as usize,
            );
        }
    }
}
