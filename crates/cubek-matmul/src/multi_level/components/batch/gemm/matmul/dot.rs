use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};
use cubecl::{prelude::*, std::tensor::ViewMut};

use crate::multi_level::components::batch::{
    CheckBounds,
    gemm::{
        PlanesSplit,
        io::{read, write},
    },
};

/// Plane-cooperative dot product over K: a block of output cells per plane.
///
/// Units within a plane share the K traversal in `plane_dim`-wide steps and
/// accumulate one `Vector<AccR, vs>` of partials per cell of the block; a final
/// horizontal (and cross-unit, when `plane_dim > 1`) sum produces each scalar to
/// write. Tile starts are swizzled by `plane_id` so concurrent planes hit K at
/// staggered offsets. When `plane_dim == 1` (CPU path) the cross-unit reduction
/// degenerates to a plain `Vector::vector_sum` and every plane writes its own
/// block.
///
/// The block runs along the axis the planes split, so the operand indexed by
/// the other axis is read once per K step for the whole block.
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
    out: ViewMut<O, Coords2d>,
    m_pos: u32,
    n_pos: u32,
    k_dim: u32,
    #[comptime] plane_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] accumulators: u32,
    #[comptime] planes_split: PlanesSplit,
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

    let mut acc = Sequence::<Vector<AccR, N>>::new();
    #[unroll]
    for _ in 0..accumulators {
        acc.push(Vector::<AccR, N>::zero());
    }
    // Rebinding as `mut` turns every accumulator into a mutable local.
    #[allow(clippy::redundant_locals)]
    let mut acc = acc;

    for tile_index in 0..num_tiles_k {
        let swizzled_tile_index = (tile_index + plane_id) % num_tiles_k;
        let k_base = swizzled_tile_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;

        match comptime!(planes_split) {
            PlanesSplit::N => {
                let lhs_val =
                    Vector::<AccR, N>::cast_from(read(&lhs, (m_pos, k_pos), check_bounds));
                #[unroll]
                for j in 0..accumulators {
                    let rhs_val = read(&rhs, (k_pos, n_pos + j), check_bounds);
                    *acc.index_mut(j as usize) += lhs_val * Vector::cast_from(rhs_val);
                }
            }
            PlanesSplit::M => {
                let rhs_val =
                    Vector::<AccR, N>::cast_from(read(&rhs, (k_pos, n_pos), check_bounds));
                #[unroll]
                for j in 0..accumulators {
                    let lhs_val = read(&lhs, (m_pos + j, k_pos), check_bounds);
                    *acc.index_mut(j as usize) += Vector::cast_from(lhs_val) * rhs_val;
                }
            }
        }
    }

    #[unroll]
    for j in 0..accumulators {
        let cell = match comptime!(planes_split) {
            PlanesSplit::N => (m_pos, n_pos + j),
            PlanesSplit::M => (m_pos + j, n_pos),
        };
        let sum = Vector::vector_sum(*acc.index(j as usize));
        if comptime!(plane_dim > 1) {
            let sum = O::cast_from(plane_sum(sum));
            if unit_id == 0 {
                write(out, cell, sum, check_bounds);
            }
        } else {
            write(out, cell, O::cast_from(sum), check_bounds);
        };
    }
}
