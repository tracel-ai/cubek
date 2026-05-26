use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};
use cubecl::{prelude::*, std::tensor::ViewMut};

use crate::components::batch::{
    CheckBounds,
    gemm::io::{read, write},
};

/// Outer-product CPU kernel covering the three non-Dot variants —
/// `OuterM` (Col-Col), `OuterNLhsContig` (Row-Row), and
/// `OuterNLhsStrided` (Col-Row) — by comptime knobs:
///
/// * `vec_axis_is_n`: which output axis the accumulator is vectorized along.
///   `true` → vec axis is N (`Vector<AccR, NR>` accumulator, vec-side is rhs,
///   K-side is lhs); `false` → vec axis is M (lhs is vec-side, rhs is K-side).
///   The "scalar axis" is the other one — held fixed per plane.
///
/// * `scalar_side_strided`: whether the K-side operand has K as a strided
///   axis (only true for `OuterNLhsStrided`: lhs is M-contig, so each K
///   position is a separate vector load and we pick this plane's lane).
///   When `false`, the K-side is K-contig and a single K-vector load per
///   tile yields `vs` scalars.
///
/// `m_pos` / `n_pos` semantics depend on `vec_axis_is_n`: the vec-axis
/// coord is the block base (incremented at write time), the scalar-axis
/// coord is the per-plane fixed position.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_outer_product<
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
    out: ViewMut<O, Coords2d>,
    m_pos: u32,
    n_pos: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] vec_axis_is_n: bool,
    #[comptime] scalar_side_strided: bool,
    #[comptime] check_bounds: CheckBounds,
) {
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;
    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles_k {
        let k_base = tile_index * vector_size;

        // Gather `vs` scalars from the K-axis side into an AccR-typed array.
        let mut scalars = Array::new(vector_size as usize);
        if comptime!(scalar_side_strided) {
            // Col-Row: lhs is M-contig (strided in K). Each read returns a
            // Vector along M; pick this plane's row by `m_pos % vs`.
            let lane = m_pos % vector_size;
            #[unroll]
            for i in 0..vector_size {
                let v = read(&lhs, (m_pos, k_base + i), check_bounds);
                scalars[i as usize] = AccR::cast_from(v.extract(lane as usize));
            }
        } else if comptime!(vec_axis_is_n) {
            // Row-Row: lhs is K-contig. One K-vec load per tile.
            let k_vec = read(&lhs, (m_pos, k_base), check_bounds);
            #[unroll]
            for i in 0..vector_size {
                scalars[i as usize] = AccR::cast_from(k_vec.extract(i as usize));
            }
        } else {
            // Col-Col: rhs is K-contig. One K-vec load per tile.
            let k_vec = read(&rhs, (k_base, n_pos), check_bounds);
            #[unroll]
            for i in 0..vector_size {
                scalars[i as usize] = AccR::cast_from(k_vec.extract(i as usize));
            }
        }

        // Per K position: load the vec-axis natural vector and broadcast-FMA.
        #[unroll]
        for i in 0..vector_size {
            let scalar_bcast = Vector::<AccR, N>::new(scalars[i as usize]);
            if comptime!(vec_axis_is_n) {
                let vec_vec = read(&rhs, (k_base + i, n_pos), check_bounds);
                acc += Vector::cast_from(vec_vec) * scalar_bcast;
            } else {
                let vec_vec = read(&lhs, (m_pos, k_base + i), check_bounds);
                acc += Vector::cast_from(vec_vec) * scalar_bcast;
            }
        }
    }

    // Write `vs` scalars along the vec axis. RowMajor output → strided
    // stores when the vec axis is M; contiguous when it's N (but
    // vector_sizes.out = 1 so we still write one scalar at a time).
    #[unroll]
    for j in 0..vector_size {
        let out_val = O::cast_from(acc.extract(j as usize));
        if comptime!(vec_axis_is_n) {
            write(out, (m_pos, n_pos + j), out_val, check_bounds);
        } else {
            write(out, (m_pos + j, n_pos), out_val, check_bounds);
        }
    }
}
