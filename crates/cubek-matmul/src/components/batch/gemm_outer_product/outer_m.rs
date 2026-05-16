use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    checked_io::{read, write},
};

/// Outer-product CPU kernel for the Col-Col case (lhs M-contig, rhs
/// K-contig). Each plane produces an `MR × 1` output block (where
/// `MR = vector_size`) at `(m_pos_base, n_pos)`. The accumulator is a
/// `Vector<AccR, MR>` along M — same axis as the natural lhs column
/// vector — so the inner step is a single FMA per K position with the
/// rhs scalar broadcast.
///
/// The final write is `MR` scalar stores spread across the strided
/// RowMajor output column; that's intrinsic to writing an M-vector into
/// a RowMajor tensor and is the cost of having M as the vectorized axis.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_outer_m<
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
    m_pos_base: u32,
    n_pos: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] check_bounds: CheckBounds,
) {
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos_base >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;
    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles_k {
        let k_base = tile_index * vector_size;

        // Load rhs as one K-vector (ColMajor → K is contig at fixed N).
        let rhs_k_vec = read(rhs, (k_base, n_pos), check_bounds);
        let mut rhs_scalars = Array::<R>::new(vector_size as usize);
        #[unroll]
        for i in 0..vector_size {
            rhs_scalars[i as usize] = rhs_k_vec[i as usize];
        }

        // Per K position: load the lhs M-vector at fixed K (one ColMajor
        // contiguous load), multiply by the broadcast RHS scalar, and
        // accumulate into the per-column accumulator along M.
        #[unroll]
        for i in 0..vector_size {
            let lhs_m_vec = read(lhs, (m_pos_base, k_base + i), check_bounds);
            let rhs_broadcast = Vector::<AccR, N>::new(AccR::cast_from(rhs_scalars[i as usize]));
            acc += Vector::cast_from(lhs_m_vec) * rhs_broadcast;
        }
    }

    // Write MR scalars down the column. RowMajor output → strided stores.
    #[unroll]
    for i in 0..vector_size {
        let out_val = O::cast_from(acc[i as usize]);
        write(out, (m_pos_base + i, n_pos), out_val, check_bounds);
    }
}
