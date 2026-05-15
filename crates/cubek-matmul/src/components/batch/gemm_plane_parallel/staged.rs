use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    gemm_plane_parallel::{
        KAccess,
        io::{read, write},
    },
};

/// Unified staged matmul kernel.
///
/// Each plane produces an `m_stride × n_stride` block of output where
/// `m_stride` (resp. `n_stride`) is `vector_size` if lhs (resp. rhs) is
/// staged and `1` otherwise. The four `(lhs_access, rhs_access)` combos
/// recover the historical paths:
///
///   - `(Direct, Direct)`: plain GEMM — 1×1 block (only used on CPU; on
///     GPU the dedicated [`super::plane::execute_plane`] is preferred
///     for its in-plane K parallelism).
///   - `(Direct, Staged)`: single-staged rhs — 1×VS block (vec×mat or
///     Row×Row mat-mat).
///   - `(Staged, Direct)`: single-staged lhs — VS×1 block (mat×vec or
///     Col×Col mat-mat).
///   - `(Staged, Staged)`: double-staged — VS×VS block (Col×Row mat-mat).
///
/// Per K-tile each side loads its K-vectors via [`load_k_vecs_lhs`] /
/// [`load_k_vecs_rhs`] (a single vector load when direct, or `vector_size`
/// reads + transpose when staged). The kernel then computes the outer
/// product of the per-row `Vector<AccR, N>` partials and accumulates into
/// `accs[i * n_stride + j]`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_staged_block<
    L: Scalar,
    R: Scalar,
    O: CubePrimitive,
    AccR: Numeric,
    SML: Scalar,
    SMR: Scalar,
    LS: Size,
    RS: Size,
    N: Size,
>(
    lhs: View<Vector<L, LS>, Coords2d>,
    rhs: View<Vector<R, RS>, Coords2d>,
    out: View<O, Coords2d, ReadWrite>,
    m_id: u32,
    n_id: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] lhs_access: KAccess,
    #[comptime] rhs_access: KAccess,
    #[comptime] check_bounds: CheckBounds,
) {
    let m_stride = comptime!(stride_for(lhs_access, vector_size));
    let n_stride = comptime!(stride_for(rhs_access, vector_size));
    let m_pos = m_id * m_stride;
    let n_pos = n_id * n_stride;

    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;

    // accs[i * n_stride + j]: partials for output cell at (m_pos+i, n_pos+j).
    let mut accs: Array<Vector<AccR, N>> = Array::new((m_stride * n_stride) as usize);
    for idx in 0..(m_stride * n_stride) {
        accs[idx as usize] = Vector::zero();
    }

    // Per-side K-vector buffers, sized to the side's stride.
    let mut lhs_k_vecs = Array::<Vector<SML, N>>::new(m_stride as usize);
    let mut rhs_k_vecs = Array::<Vector<SMR, N>>::new(n_stride as usize);

    // Shared-memory staging buffers, hoisted out of the per-K-tile loop
    // so they're allocated once per plane (and only when actually used).
    // Sized to 1 when the side is direct — cube needs a non-zero array.
    let lhs_tile_size = comptime!(tile_buffer_size(lhs_access, vector_size));
    let rhs_tile_size = comptime!(tile_buffer_size(rhs_access, vector_size));
    let mut lhs_tile = Array::<SML>::new(lhs_tile_size as usize);
    let mut rhs_tile = Array::<SMR>::new(rhs_tile_size as usize);

    for tile_index in 0..num_tiles_k {
        let k_base = tile_index * vector_size;

        // lhs view is indexed `(m, k)`; rhs is `(k, n)`. The loader
        // baked in the per-side layout so callers don't repeat that.
        load_k_vecs_lhs::<L, LS, SML, N>(
            lhs,
            m_pos,
            k_base,
            &mut lhs_k_vecs,
            &mut lhs_tile,
            vector_size,
            lhs_access,
            check_bounds,
        );
        load_k_vecs_rhs::<R, RS, SMR, N>(
            rhs,
            n_pos,
            k_base,
            &mut rhs_k_vecs,
            &mut rhs_tile,
            vector_size,
            rhs_access,
            check_bounds,
        );

        // Outer-product accumulate: each (i, j) cell adds a `Vector<AccR, N>`
        // partial product across the `vector_size` K-positions in this tile.
        #[unroll]
        for i in 0..m_stride {
            let lhs_k_acc: Vector<AccR, N> = Vector::cast_from(lhs_k_vecs[i as usize]);
            #[unroll]
            for j in 0..n_stride {
                accs[(i * n_stride + j) as usize] +=
                    lhs_k_acc * Vector::cast_from(rhs_k_vecs[j as usize]);
            }
        }
    }

    // Reduce each per-cell partial-K accumulator and write.
    for i in 0..m_stride {
        for j in 0..n_stride {
            let acc = accs[(i * n_stride + j) as usize];
            let sum = O::cast_from(Vector::vector_sum(acc));
            write(out, (m_pos + i, n_pos + j), sum, check_bounds);
        }
    }
}

fn stride_for(access: KAccess, vector_size: u32) -> u32 {
    match access {
        KAccess::Direct => 1,
        KAccess::Staged => vector_size,
    }
}

fn tile_buffer_size(access: KAccess, vector_size: u32) -> u32 {
    match access {
        // Direct sides don't stage; allocate the smallest non-zero buffer
        // cube tolerates rather than wasting `vector_size²` slots.
        KAccess::Direct => 1,
        KAccess::Staged => vector_size * vector_size,
    }
}

/// Fill `out` with the per-row K-vectors of `lhs` for the current K-tile.
///
///   - `Direct`: lhs is K-contiguous (RowMajor or Vector). One vector load
///     at `(m_pos, k_base)` yields the K-vector for the single row — `out`
///     has length 1.
///   - `Staged`: lhs is ColMajor. `vector_size` vector loads at
///     `(m_pos, k_base + seg)` each return `vector_size` M-elements at
///     fixed K. After load, transpose-extract one K-vector per M-row —
///     `out` has length `vector_size`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_k_vecs_lhs<L: Scalar, LS: Size, SM: Scalar, N: Size>(
    lhs: View<Vector<L, LS>, Coords2d>,
    m_pos: u32,
    k_base: u32,
    out: &mut Array<Vector<SM, N>>,
    tile: &mut Array<SM>,
    #[comptime] vector_size: u32,
    #[comptime] access: KAccess,
    #[comptime] check_bounds: CheckBounds,
) {
    match comptime!(access) {
        KAccess::Direct => {
            let v = read(lhs, (m_pos, k_base), check_bounds);
            out[0] = Vector::cast_from(v);
        }
        KAccess::Staged => {
            for seg in 0..vector_size {
                let v = read(lhs, (m_pos, k_base + seg), check_bounds);
                #[unroll]
                for i in 0..vector_size {
                    tile[(seg * vector_size + i) as usize] = SM::cast_from(v[i as usize]);
                }
            }
            #[unroll]
            for i in 0..vector_size {
                let mut k_vec: Vector<SM, N> = Vector::empty();
                #[unroll]
                for seg in 0..vector_size {
                    k_vec[seg as usize] = tile[(seg * vector_size + i) as usize];
                }
                out[i as usize] = k_vec;
            }
        }
    }
}

/// Mirror of [`load_k_vecs_lhs`] for the rhs operand (indexed `(k, n)`).
///
///   - `Direct`: rhs is K-contiguous (ColMajor or Vector). One vector load
///     at `(k_base, n_pos)` yields the K-vector for the single column.
///   - `Staged`: rhs is RowMajor. `vector_size` loads at `(k_base+seg, n_pos)`
///     each return `vector_size` N-elements; transpose to get per-column
///     K-vectors.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_k_vecs_rhs<R: Scalar, RS: Size, SM: Scalar, N: Size>(
    rhs: View<Vector<R, RS>, Coords2d>,
    n_pos: u32,
    k_base: u32,
    out: &mut Array<Vector<SM, N>>,
    tile: &mut Array<SM>,
    #[comptime] vector_size: u32,
    #[comptime] access: KAccess,
    #[comptime] check_bounds: CheckBounds,
) {
    match comptime!(access) {
        KAccess::Direct => {
            let v = read(rhs, (k_base, n_pos), check_bounds);
            out[0] = Vector::cast_from(v);
        }
        KAccess::Staged => {
            for seg in 0..vector_size {
                let v = read(rhs, (k_base + seg, n_pos), check_bounds);
                #[unroll]
                for j in 0..vector_size {
                    tile[(seg * vector_size + j) as usize] = SM::cast_from(v[j as usize]);
                }
            }
            #[unroll]
            for j in 0..vector_size {
                let mut k_vec: Vector<SM, N> = Vector::empty();
                #[unroll]
                for seg in 0..vector_size {
                    k_vec[seg as usize] = tile[(seg * vector_size + j) as usize];
                }
                out[j as usize] = k_vec;
            }
        }
    }
}
