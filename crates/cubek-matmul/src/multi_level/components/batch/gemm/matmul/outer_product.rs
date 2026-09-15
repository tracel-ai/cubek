use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};
use cubecl::{prelude::*, std::tensor::ViewMut};

use crate::multi_level::components::batch::{
    CheckBounds,
    gemm::io::{read, write},
};

/// Outer-product CPU kernel covering the two non-Dot variants,
/// `OuterM` (Col-Col) and `OuterN` (Row-Row): by one comptime knob:
///
/// * `vec_axis_is_n`: which output axis the accumulators are vectorized along.
///   `true` → vec axis is N (`Vector<AccR, NR>` accumulators, vec-side is rhs,
///   K-side is lhs); `false` → vec axis is M (lhs is vec-side, rhs is K-side).
///   The "scalar axis" is the other one: held fixed per plane.
///
/// The K-side operand is K-contig either way, so one K-vector load per tile
/// yields the `vs` scalars, and each scalar serves every accumulator of the block.
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
    #[comptime] accumulators: u32,
    #[comptime] vec_axis_is_n: bool,
    #[comptime] check_bounds: CheckBounds,
) {
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;

    let mut acc = Sequence::<Vector<AccR, N>>::new();
    #[unroll]
    for _ in 0..accumulators {
        acc.push(Vector::<AccR, N>::zero());
    }
    // Rebinding as `mut` turns every accumulator into a mutable local.
    #[allow(clippy::redundant_locals)]
    let mut acc = acc;

    for tile_index in 0..num_tiles_k {
        let k_base = tile_index * vector_size;

        // Gather `vs` scalars from the K-axis side into an AccR-typed array.
        let mut scalars = Array::new(vector_size as usize);
        if comptime!(vec_axis_is_n) {
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

        // Per K position: load each accumulator's natural vector and broadcast-FMA.
        #[unroll]
        for i in 0..vector_size {
            let scalar_bcast = Vector::<AccR, N>::new(scalars[i as usize]);
            #[unroll]
            for j in 0..accumulators {
                let block_offset = j * vector_size;
                if comptime!(vec_axis_is_n) {
                    let vec_vec = read(&rhs, (k_base + i, n_pos + block_offset), check_bounds);
                    *acc.index_mut(j as usize) += Vector::cast_from(vec_vec) * scalar_bcast;
                } else {
                    let vec_vec = read(&lhs, (m_pos + block_offset, k_base + i), check_bounds);
                    *acc.index_mut(j as usize) += Vector::cast_from(vec_vec) * scalar_bcast;
                }
            }
        }
    }

    // Write `vs` scalars per accumulator along the vec axis. RowMajor output →
    // strided stores when the vec axis is M; contiguous when it's N (but
    // vector_sizes.out = 1 so we still write one scalar at a time).
    #[unroll]
    for j in 0..accumulators {
        let lanes = acc[j as usize];
        #[unroll]
        for l in 0..vector_size {
            let offset = j * vector_size + l;
            let out_val = O::cast_from(lanes.extract(l as usize));
            if comptime!(vec_axis_is_n) {
                write(out, (m_pos, n_pos + offset), out_val, check_bounds);
            } else {
                write(out, (m_pos + offset, n_pos), out_val, check_bounds);
            }
        }
    }
}
