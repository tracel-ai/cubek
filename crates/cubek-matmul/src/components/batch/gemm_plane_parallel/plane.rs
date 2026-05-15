use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    gemm_plane_parallel::io::{read, write},
};

/// Plane-cooperative GEMM kernel. K is contiguous on both operands and
/// the `plane_dim` units within a plane each chew through their slice of
/// `num_segments_k` K-segments; `plane_sum` reduces across units and the
/// leader unit writes. Only valid for `plane_dim > 1` (GPU backends);
/// the staged kernel handles CPU.
///
/// `(m_pos, n_pos)` is the output cell this plane owns — already
/// resolved by the caller from the cube grid, planes-split, and `plane_id`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_plane<
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
    #[comptime] plane_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] check_bounds: CheckBounds,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let segment_size = plane_dim * vector_size;
    let num_segments_k = k_dim / segment_size;

    let mut acc = Vector::<AccR, N>::zero();

    for segment_index in 0..num_segments_k {
        let swizzled_segment_index = (segment_index + plane_id) % num_segments_k;
        let k_base = swizzled_segment_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;

        let lhs_val = read(lhs, (m_pos, k_pos), check_bounds);
        let rhs_val = read(rhs, (k_pos, n_pos), check_bounds);

        acc += Vector::cast_from(lhs_val) * Vector::cast_from(rhs_val);
    }

    let sum = O::cast_from(plane_sum(Vector::vector_sum(acc)));

    if unit_id == 0 {
        write(out, (m_pos, n_pos), sum, check_bounds);
    }
}
