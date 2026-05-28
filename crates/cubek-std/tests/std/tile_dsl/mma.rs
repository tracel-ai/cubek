//! Lowering for the whole-tensor `mma`: the gmem-accumulator walk, the matmul's
//! tile-grid gathering, the scalar leaf matmul, and the element copy.

use cubecl::{
    prelude::*,
    std::tensor::{
        AsViewMut, AsViewMutExpand, ViewMut,
        layout::{Coords2d, CoordsDyn},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// Accumulator in global memory. Walks the partitioner; each step stages both
/// operand leaves into shared memory and accumulates the product into the output
/// leaf.
#[cube]
pub fn mma_gmem<E: Numeric, S: Size>(
    out: &Tile<'_, E, S, CoordsDyn>,
    lhs: &Tile<'_, E, S, CoordsDyn>,
    rhs: &Tile<'_, E, S, CoordsDyn>,
) {
    // The operation ranges over the union of its operands' spaces
    // ({M,N} ∪ {M,K} ∪ {K,N} = {M,N,K}) and contracts the axes the output drops.
    let space = comptime!(Space::union(&[&out.space, &lhs.space, &rhs.space]));
    let contracted = comptime!(space.contracting(&out.space));
    comptime!(assert!(
        !contracted.is_empty(),
        "mma: the output must drop at least one (contracted) axis"
    ));

    // Stage each operand at its own axes' sub-tile size (comptime).
    let a_rows = lhs.partitioner.sub_tile_edge(comptime!(lhs.space.axis_at(0)));
    let a_cols = lhs.partitioner.sub_tile_edge(comptime!(lhs.space.axis_at(1)));
    let b_rows = rhs.partitioner.sub_tile_edge(comptime!(rhs.space.axis_at(0)));
    let b_cols = rhs.partitioner.sub_tile_edge(comptime!(rhs.space.axis_at(1)));

    let mut a_smem = Shared::<[Vector<E, S>]>::new_slice(comptime!((a_rows * a_cols) as usize));
    let mut b_smem = Shared::<[Vector<E, S>]>::new_slice(comptime!((b_rows * b_cols) as usize));
    let mut a_tile = stage_smem::<E, S>(
        a_smem.view_mut(smem_tile_layout(a_rows, a_cols)),
        comptime!(lhs.space.clone()),
        lhs.partitioner.clone(),
    );
    let mut b_tile = stage_smem::<E, S>(
        b_smem.view_mut(smem_tile_layout(b_rows, b_cols)),
        comptime!(rhs.space.clone()),
        rhs.partitioner.clone(),
    );

    // The matmul's tile grid (gathered from the operands), walked by the
    // partitioner.
    let grid = mma_grid::<E, S>(out, lhs, rhs, comptime!(space.clone()));
    let walk = out.partitioner.walk(grid);
    let total = walk.total();
    for i in 0..total {
        let point = walk.point(i);

        let a_leaf = lhs.partition(&point);
        let b_leaf = rhs.partition(&point);
        let mut acc = out.partition(&point);

        a_tile.copy_from(&a_leaf);
        b_tile.copy_from(&b_leaf);
        acc.mma(&a_tile, &b_tile);
    }
}

/// This matmul's tile [`Grid`] for `space`: each axis's tile count read from an
/// operand that carries it. The partitioner takes the grid from here.
#[cube]
fn mma_grid<E: Numeric, S: Size>(
    out: &Tile<'_, E, S, CoordsDyn>,
    lhs: &Tile<'_, E, S, CoordsDyn>,
    rhs: &Tile<'_, E, S, CoordsDyn>,
    #[comptime] space: Space,
) -> Grid {
    let mut counts = Sequence::<usize>::new();
    #[unroll]
    for p in 0..comptime!(space.rank()) {
        counts.push(tiles_of::<E, S>(out, lhs, rhs, comptime!(space.axis_at(p))));
    }
    Grid::new(counts, space)
}

/// The runtime tile count along `axis`, read from whichever operand carries it.
/// Every union axis is in at least one operand.
#[cube]
fn tiles_of<E: Numeric, S: Size>(
    out: &Tile<'_, E, S, CoordsDyn>,
    lhs: &Tile<'_, E, S, CoordsDyn>,
    rhs: &Tile<'_, E, S, CoordsDyn>,
    #[comptime] axis: Axis,
) -> usize {
    if comptime!(out.space.contains(axis)) {
        out.tiles(axis)
    } else if comptime!(lhs.space.contains(axis)) {
        lhs.tiles(axis)
    } else {
        rhs.tiles(axis)
    }
}

/// Scalar 2-D contraction `acc(i, j) += Σ_c lhs(i, c) · rhs(c, j)`, shapes read
/// from the views.
#[cube]
pub fn mma_smem<E: Numeric, S: Size>(
    acc: &mut ViewMut<'_, Vector<E, S>, Coords2d>,
    lhs: &ViewMut<'_, Vector<E, S>, Coords2d>,
    rhs: &ViewMut<'_, Vector<E, S>, Coords2d>,
) {
    let (m, k) = lhs.shape();
    let (_, n) = rhs.shape();

    for i in 0..m {
        for j in 0..n {
            let mut value = acc.read((i, j));
            for c in 0..k {
                value += lhs.read((i, c)) * rhs.read((c, j));
            }
            acc.write((i, j), value);
        }
    }
}

/// Element-wise copy of `src` into `dst` (same 2-D shape).
#[cube]
pub fn copy_2d<E: Numeric, S: Size>(
    dst: &mut ViewMut<'_, Vector<E, S>, Coords2d>,
    src: &ViewMut<'_, Vector<E, S>, Coords2d>,
) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            dst.write((i, j), src.read((i, j)));
        }
    }
}
