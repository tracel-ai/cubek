//! Lowering strategies that `mma`/`copy_from` dispatch to: the gmem-accumulator
//! walk, point resolution, the scalar leaf matmul, and the element copies.

use cubecl::{
    prelude::*,
    std::tensor::{AsViewMut, AsViewMutExpand, View, layout::Coords2d},
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// Accumulator in global memory. Walks the partitioner, staging both operand
/// leaves into shared memory and accumulating the product into the output
/// (→ [`mma_leaf`]).
#[cube]
pub fn mma_gmem<E: Numeric, S: Size>(out: &Tile<E, S>, lhs: &Tile<E, S>, rhs: &Tile<E, S>) {
    // The operation ranges over the union of its operands' spaces
    // ({M,N} ∪ {M,K} ∪ {K,N} = {M,N,K}) and contracts the axes the output drops
    // (union ∖ out = {K}). `out.partition` ignores those, so each contracted fiber
    // accumulates into one output cell.
    let space = comptime!(Space::union(&[&out.space, &lhs.space, &rhs.space]));
    let contracted = comptime!(space.contracting(&out.space));
    comptime!(assert!(
        !contracted.is_empty(),
        "mma: the output must drop at least one (contracted) axis"
    ));

    // Stage each operand at its own axes' sub-tile size (comptime, from the
    // partitioner).
    let a_rows = lhs.partitioner.sub_tile_edge(comptime!(lhs.space.axis_at(0)));
    let a_cols = lhs.partitioner.sub_tile_edge(comptime!(lhs.space.axis_at(1)));
    let b_rows = rhs.partitioner.sub_tile_edge(comptime!(rhs.space.axis_at(0)));
    let b_cols = rhs.partitioner.sub_tile_edge(comptime!(rhs.space.axis_at(1)));

    let mut a_smem = SharedMemory::<Vector<E, S>>::new(comptime!((a_rows * a_cols) as usize));
    let mut b_smem = SharedMemory::<Vector<E, S>>::new(comptime!((b_rows * b_cols) as usize));
    let a_tile = Smem::stage::<E, S>(
        a_smem.view_mut(smem_tile_layout(a_rows, a_cols)),
        comptime!(lhs.space.clone()),
        lhs.partitioner.clone(),
    );
    let b_tile = Smem::stage::<E, S>(
        b_smem.view_mut(smem_tile_layout(b_rows, b_cols)),
        comptime!(rhs.space.clone()),
        rhs.partitioner.clone(),
    );

    // Build this matmul's tile grid from the operands, then let the partitioner
    // walk it. The grid is the matmul-specific half; the walk (order, iteration)
    // is generic, so no index arithmetic leaks here.
    let grid = mma_grid::<E, S>(out, lhs, rhs, comptime!(space.clone()));
    let walk = out.partitioner.walk(grid);
    let total = walk.total();
    for i in 0..total {
        let point = walk.point(i);

        let a_leaf = lhs.partition(&point);
        let b_leaf = rhs.partition(&point);
        let acc = out.partition(&point);

        a_tile.copy_from(&a_leaf);
        b_tile.copy_from(&b_leaf);
        acc.mma(&a_tile, &b_tile);
    }
}

/// This matmul's tile [`Grid`] for `space`, each axis's tile count read from an
/// operand that carries it. The matmul-specific half of building a walk; the
/// partitioner takes the grid from here ([`Partitioner::walk`]).
#[cube]
fn mma_grid<E: Numeric, S: Size>(
    out: &Tile<E, S>,
    lhs: &Tile<E, S>,
    rhs: &Tile<E, S>,
    #[comptime] space: Space,
) -> Grid {
    let mut counts = Sequence::<u32>::new();
    #[unroll]
    for p in 0..comptime!(space.rank()) {
        counts.push(tiles_of::<E, S>(out, lhs, rhs, comptime!(space.axis_at(p))));
    }
    Grid::new(counts, space)
}

/// The runtime tile count along `axis`, read from whichever operand carries it
/// (comptime choice, runtime read). Every union axis is in at least one operand.
#[cube]
fn tiles_of<E: Numeric, S: Size>(
    out: &Tile<E, S>,
    lhs: &Tile<E, S>,
    rhs: &Tile<E, S>,
    #[comptime] axis: Axis,
) -> u32 {
    if comptime!(out.space.contains(axis)) {
        out.tiles(axis)
    } else if comptime!(lhs.space.contains(axis)) {
        lhs.tiles(axis)
    } else {
        rhs.tiles(axis)
    }
}

/// Leaf accumulator (shared memory or the output tile). Lowers to [`mma_smem`].
#[cube]
pub fn mma_leaf<E: Numeric, S: Size>(acc: &Tile<E, S>, lhs: &Tile<E, S>, rhs: &Tile<E, S>) {
    match (&lhs.kind, &rhs.kind, &acc.kind) {
        (TileKind::Smem(l), TileKind::Smem(r), TileKind::Smem(a)) => mma_smem::<E, S>(a, l, r),
        (TileKind::Smem(l), TileKind::Smem(r), TileKind::GmemWrite(a)) => mma_smem::<E, S>(a, l, r),
        _ => panic!("mma_leaf: operands must be shared-memory tiles"),
    }
}

/// Scalar 2-D contraction `acc(i, j) += Σ_c lhs(i, c) · rhs(c, j)`, shapes read
/// from the views.
#[cube]
fn mma_smem<E: Numeric, S: Size>(
    acc: &View<Vector<E, S>, Coords2d, ReadWrite>,
    lhs: &View<Vector<E, S>, Coords2d, ReadWrite>,
    rhs: &View<Vector<E, S>, Coords2d, ReadWrite>,
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

#[cube]
pub fn smem_load_from<E: Numeric, S: Size>(
    dst: &View<Vector<E, S>, Coords2d, ReadWrite>,
    src: &Tile<E, S>,
) {
    match &src.kind {
        TileKind::GmemRead(s) => copy_2d_ro::<E, S>(dst, s),
        TileKind::GmemTiledRead(_)
        | TileKind::GmemTiledWrite(_)
        | TileKind::GmemWrite(_)
        | TileKind::Smem(_) => panic!("smem load: source must be a gmem read tile"),
    }
}

#[cube]
pub fn gmem_store_from<E: Numeric, S: Size>(
    dst: &View<Vector<E, S>, Coords2d, ReadWrite>,
    src: &Tile<E, S>,
) {
    match &src.kind {
        TileKind::Smem(s) => copy_2d_rw::<E, S>(dst, s),
        TileKind::GmemTiledRead(_)
        | TileKind::GmemTiledWrite(_)
        | TileKind::GmemRead(_)
        | TileKind::GmemWrite(_) => panic!("gmem store: source must be an smem tile"),
    }
}

#[cube]
fn copy_2d_ro<E: Numeric, S: Size>(
    dst: &View<Vector<E, S>, Coords2d, ReadWrite>,
    src: &View<Vector<E, S>, Coords2d>,
) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            dst.write((i, j), src.read((i, j)));
        }
    }
}

#[cube]
fn copy_2d_rw<E: Numeric, S: Size>(
    dst: &View<Vector<E, S>, Coords2d, ReadWrite>,
    src: &View<Vector<E, S>, Coords2d, ReadWrite>,
) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            dst.write((i, j), src.read((i, j)));
        }
    }
}
