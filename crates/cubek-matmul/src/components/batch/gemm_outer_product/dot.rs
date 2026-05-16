use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    checked_io::{read, write},
};

/// Dot-product CPU kernel for the Row-Col case (K-contiguous on both sides).
/// One output cell per plane; the K loop accumulates a `Vector<AccR, vs>`
/// of partials and a final horizontal sum produces the scalar to write.
///
/// Mirrors the gemm_plane_parallel design (same algorithm) but lives here
/// so the outer-product routine is self-contained on the Row-Col baseline.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_dot<
    L: CubePrimitive,
    R: CubePrimitive,
    O: CubePrimitive,
    AccR: Numeric,
    N: Size,
>(
    lhs: View<L, Coords2d>,
    rhs: View<R, Coords2d>,
    out: View<O, Coords2d, ReadWrite>,
    m_pos: u32,
    n_pos: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
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
        let k_pos = tile_index * vector_size;
        let lhs_val = read(lhs, (m_pos, k_pos), check_bounds);
        let rhs_val = read(rhs, (k_pos, n_pos), check_bounds);
        acc += Vector::cast_from(lhs_val) * Vector::cast_from(rhs_val);
    }

    let sum = O::cast_from(Vector::vector_sum(acc));
    write(out, (m_pos, n_pos), sum, check_bounds);
}
