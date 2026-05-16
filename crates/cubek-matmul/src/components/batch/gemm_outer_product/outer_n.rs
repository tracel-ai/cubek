use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    checked_io::{read, write},
};

/// Outer-product CPU kernel for the cases where rhs is N-contig (RowMajor).
/// Each plane produces a `1 × NR` output block (where `NR = vector_size`)
/// at `(m_pos, n_pos_base)`. The accumulator is a `Vector<AccR, NR>` along
/// N — same axis as the natural rhs row vector — so the inner step is a
/// single broadcast-FMA per K position with no transpose.
///
/// Two LHS access shapes are supported, dispatched at comptime:
///   - `lhs_k_contig = true` (Row-Row): one vector load per K-tile fetches
///     all `vs` LHS scalars at once; the inner loop reads them by lane.
///   - `lhs_k_contig = false` (Col-Row): LHS is M-contig, so each K
///     position is a separate scalar load (strided in K).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_outer_n<
    L: Scalar,
    R: Scalar,
    O: CubePrimitive,
    AccR: Numeric,
    LS: Size,
    RS: Size,
    N: Size,
>(
    lhs: View<Vector<L, LS>, Coords2d>,
    rhs: View<Vector<R, RS>, Coords2d>,
    out: View<O, Coords2d, ReadWrite>,
    m_pos: u32,
    n_pos_base: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] lhs_k_contig: bool,
    #[comptime] check_bounds: CheckBounds,
) {
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos_base >= out_n {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;
    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles_k {
        let k_base = tile_index * vector_size;

        // Pull `vs` LHS scalars covering K positions `[k_base, k_base+vs)`.
        let mut lhs_scalars = Array::<L>::new(vector_size as usize);
        if comptime!(lhs_k_contig) {
            // Row-Row: lhs is K-contig, one vector load per K-tile.
            let lhs_k_vec = read(lhs, (m_pos, k_base), check_bounds);
            #[unroll]
            for i in 0..vector_size {
                lhs_scalars[i as usize] = lhs_k_vec[i as usize];
            }
        } else {
            // Col-Row: lhs is M-contig (strided in K). Each vector load
            // returns `vs` M-elements at fixed K, aligned to the M-block
            // containing `m_pos`; index by `m_pos % vs` to pick this
            // plane's row out of the block.
            let lane = m_pos % vector_size;
            #[unroll]
            for i in 0..vector_size {
                let v = read(lhs, (m_pos, k_base + i), check_bounds);
                lhs_scalars[i as usize] = v[lane as usize];
            }
        }

        // Broadcast-FMA per K position: each rhs N-vector is a single
        // contig row of NR n-elements; multiply by the broadcast LHS
        // scalar and accumulate into the per-row accumulator.
        #[unroll]
        for i in 0..vector_size {
            let rhs_n_vec = read(rhs, (k_base + i, n_pos_base), check_bounds);
            let lhs_broadcast = Vector::<AccR, N>::new(AccR::cast_from(lhs_scalars[i as usize]));
            acc += lhs_broadcast * Vector::cast_from(rhs_n_vec);
        }
    }

    // Write each lane of the accumulator as a scalar (out is RowMajor;
    // the NR cells are contiguous in memory, but the kernel writes one
    // scalar at a time since vector_sizes.out = 1).
    #[unroll]
    for j in 0..vector_size {
        let out_val = O::cast_from(acc[j as usize]);
        write(out, (m_pos, n_pos_base + j), out_val, check_bounds);
    }
}
