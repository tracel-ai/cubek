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
    let partitioner = comptime!(out.partitioner.clone());

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

    // Stage each operand at its own axes' sub-tile size.
    let a_rows = comptime!(partitioner.sub_tile_edge(lhs.space.axis_at(0)));
    let a_cols = comptime!(partitioner.sub_tile_edge(lhs.space.axis_at(1)));
    let b_rows = comptime!(partitioner.sub_tile_edge(rhs.space.axis_at(0)));
    let b_cols = comptime!(partitioner.sub_tile_edge(rhs.space.axis_at(1)));

    let mut a_smem = SharedMemory::<Vector<E, S>>::new(comptime!((a_rows * a_cols) as usize));
    let mut b_smem = SharedMemory::<Vector<E, S>>::new(comptime!((b_rows * b_cols) as usize));
    let a_tile = Smem::stage::<E, S>(
        a_smem.view_mut(smem_tile_layout(a_rows, a_cols)),
        comptime!(lhs.space.clone()),
        partitioner.clone(),
    );
    let b_tile = Smem::stage::<E, S>(
        b_smem.view_mut(smem_tile_layout(b_rows, b_cols)),
        comptime!(rhs.space.clone()),
        partitioner.clone(),
    );

    let walk = comptime!(partitioner.walk(&space));
    let steps = comptime!(walk.len() as u32);
    #[unroll]
    for i in 0..steps {
        let point = resolve(comptime!(walk[i as usize].clone()), space.clone());

        let a_leaf = lhs.partition(&point);
        let b_leaf = rhs.partition(&point);
        let acc = out.partition(&point);

        a_tile.copy_from(&a_leaf);
        b_tile.copy_from(&b_leaf);
        acc.mma(&a_tile, &b_tile);
    }
}

/// Resolve a walk [`Cell`] to a runtime [`Point`]: one index per axis in the
/// space's order, tagged with that space as the point's frame.
#[cube]
fn resolve(#[comptime] cell: Cell, #[comptime] space: Space) -> Point {
    let mut coords = Sequence::<u32>::new();
    #[unroll]
    for i in 0..comptime!(space.rank()) {
        let coord = comptime!(cell.get(space.axis_at(i)));
        coords.push(axis_pos(coord, comptime!(coord.base())));
    }
    Point::new(coords, comptime!(space.clone()))
}

/// One axis of [`resolve`]: `base` (the `Fixed` index or `Affine` offset) seeds a
/// runtime value; an `Affine` axis adds its hardware term `hw_pos · stride`.
#[cube]
fn axis_pos(#[comptime] coord: Coord, base: u32) -> u32 {
    let mut pos = base;
    if comptime!(coord.is_affine()) {
        pos = hw_pos(comptime!(coord.unit())) * comptime!(coord.stride()) + base;
    }
    pos
}

/// This cube's position along the dimension a `Cube` primitive rides. (Plane and
/// Unit spreading land with the inner levels.)
#[cube]
fn hw_pos(#[comptime] unit: ComputePrimitive) -> u32 {
    #[comptime]
    let dim = match unit {
        ComputePrimitive::Cube(dim) => dim,
        _ => {
            panic!("hw_pos: only Cube spreading is implemented (Plane/Unit are inner-level seams)")
        }
    };
    // Comptime selection: only the matched arm's builtin is emitted.
    let mut pos = CUBE_POS_X;
    if comptime!(matches!(dim, CubeDimension::Y)) {
        pos = CUBE_POS_Y;
    }
    if comptime!(matches!(dim, CubeDimension::Z)) {
        pos = CUBE_POS_Z;
    }
    pos
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
