use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};

use crate::components::batch::{
    CheckBounds,
    gemm::io::{read, write},
};

/// Plane-cooperative dot product over K — one output cell per plane.
///
/// Units within a plane share the K traversal in `plane_dim`-wide steps and
/// accumulate a `Vector<AccR, vs>` of partials; a final horizontal (and
/// cross-unit, when `plane_dim > 1`) sum produces the scalar to write. Tile
/// starts are swizzled by `plane_id` so concurrent planes hit K at staggered
/// offsets. When `plane_dim == 1` (CPU path) the cross-unit reduction
/// degenerates to a plain `Vector::vector_sum` and every plane writes its
/// own cell.
///
/// Layout precondition: lhs is row-major [M, K], rhs is col-major [K, N]
/// (i.e. K is the contiguous axis on both operands).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dot<
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

    let tile_size = plane_dim * vector_size;
    let num_tiles_k = k_dim / tile_size;

    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles_k {
        let swizzled_tile_index = (tile_index + plane_id) % num_tiles_k;
        let k_base = swizzled_tile_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;

        let lhs_val = read(&lhs, (m_pos, k_pos), check_bounds);
        let rhs_val = read(&rhs, (k_pos, n_pos), check_bounds);

        acc += Vector::cast_from(lhs_val) * Vector::cast_from(rhs_val);
    }

    if comptime!(plane_dim > 1) {
        let sum = O::cast_from(plane_sum(Vector::vector_sum(acc)));
        if unit_id == 0 {
            write(&out, (m_pos, n_pos), sum, check_bounds);
        }
    } else {
        let sum = O::cast_from(Vector::vector_sum(acc));
        write(&out, (m_pos, n_pos), sum, check_bounds);
    };
}
