//! The [`Tile`]: a view tagged with its [`Space`] and [`Partitioner`]. The
//! launchable unit — kernels take tiles directly.
//!
//! The arity-agnostic half: `CoordsDyn` whole tensors and the
//! [`partition`](Tile::partition) seam that collapses them into `Coords2d`
//! leaves. A tile's last two axes are its matrix (leaf) axes; any *leading*
//! axes are batch axes that `partition` pins to the walk point — so a 2-D
//! operation and its batched form share one code path. The 2-D leaf machinery
//! (including [`Tile2d`](super::Tile2d)) lives in [`dim2`](super::dim2).

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coordinates, CoordsDyn, Layout, LayoutExpand},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// What memory a tile lives in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TileKind {
    /// A whole global tensor — a launch input or the accumulator.
    GmemWhole,
    /// One tile of a global tensor — a `partition` leaf.
    GmemLeaf,
    /// A shared-memory tile — staging buffer or leaf accumulator.
    Smem,
}

/// A view + its [`Space`] + [`Partitioner`], generic over the view's coordinate
/// type `C`. You rarely name this directly: [`Tile`] is the whole-tensor /
/// arbitrary-rank form (`C = CoordsDyn`) — the launchable unit and the only kind
/// that carries [`partition`](Tile::partition)/[`tiles`](Tile::tiles)/[`mma`](Tile::mma) —
/// and [`Tile2d`](super::Tile2d) is the 2-D leaf (`C = Coords2d`).
#[derive(CubeType, CubeLaunch)]
pub struct TileOf<'a, E: Numeric, S: Size, C: Coordinates + 'a> {
    pub view: ViewMut<'a, Vector<E, S>, C>,
    pub partitioner: Partitioner,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub kind: TileKind,
}

/// A whole-tensor / arbitrary-rank tile — the default, launchable form.
/// `Tile<'_, E, S>` is all you write; `C` is `CoordsDyn`.
pub type Tile<'a, E, S> = TileOf<'a, E, S, CoordsDyn>;

/// Launch argument for a [`Tile`] (or any [`TileOf`]) — a passthrough to the
/// `CubeLaunch`-generated companion, so `TileLaunch::new(view, …)` reads
/// naturally.
pub type TileLaunch<'a, E, S, C, R> = TileOfLaunch<'a, E, S, C, R>;

#[cube]
impl<'a, E: Numeric, S: Size> TileOf<'a, E, S, CoordsDyn> {
    /// The sub-tile at grid `point`: every axis windowed to its sub-tile edge,
    /// at the origin `grid_index × edge`. The result is a same-rank [`Tile`]
    /// leaf — `partition` is *not* opinionated about which axes are matrix vs
    /// batch. Collapsing to a 2-D matrix is a separate step ([`matrix`]); a leaf
    /// with a wide leading (batch) axis can instead be vectorized along it. Each
    /// tile reads the point along its *own* axes, so operands and accumulator
    /// match even in different spaces.
    pub fn partition(&self, point: &Point) -> Tile<'a, E, S> {
        // The leaf's kind follows the source's: carving a sub-tile of shared
        // memory stays [`Smem`](TileKind::Smem); carving a global tensor (whole
        // or an already-partitioned leaf, for multi-level tiling) yields a
        // [`GmemLeaf`](TileKind::GmemLeaf).
        let kind = match comptime!(self.kind) {
            TileKind::Smem => comptime!(TileKind::Smem),
            _ => comptime!(TileKind::GmemLeaf),
        };

        let mut origin = CoordsDyn::new();
        let mut extent = CoordsDyn::new();
        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            let axis = comptime!(self.space.axis_at(p));
            let edge = self.partitioner.sub_tile_edge(axis);
            origin.push((point.get(axis) * edge) as u32);
            extent.push(comptime!(edge as i64) as u32);
        }
        let layout = BlockWindow::new(origin, extent);
        TileOf::<'a, E, S, CoordsDyn> {
            view: self.view.clone().view_mut(layout),
            partitioner: self.partitioner.clone(),
            space: comptime!(self.space.clone()),
            kind: comptime!(kind),
        }
    }

    /// Number of tiles along `axis`: extent / sub-tile size.
    pub fn tiles(&self, #[comptime] axis: Axis) -> usize {
        let shape = self.view.shape();
        let extent = *shape.index(comptime!(self.space.position(axis))) as usize;
        extent / self.partitioner.sub_tile_edge(axis)
    }

    /// Accumulate `lhs · rhs` into `self` — the tile-DSL `mma` primitive, driven
    /// by the accumulator and dispatched on its [`TileKind`].
    ///
    /// The operation ranges over the union of its operands' spaces and contracts
    /// the axes the output drops. For a batched matmul that's `{B,M,K} ∪ {B,K,N}
    /// ∪ {B,M,N} = {B,M,N,K}`, contracting `{K}`; the batch axis `B` is carried
    /// by the output, so it is not contracted.
    ///
    /// - [`GmemWhole`](TileKind::GmemWhole): walk `self`'s partitioner; each step
    ///   partitions all three operands to their leaves and *recurses* into the
    ///   leaf mma.
    /// - leaf ([`GmemLeaf`](TileKind::GmemLeaf)/[`Smem`](TileKind::Smem)): the
    ///   base case — contract each 2-D [`matrix`](Tile::matrix) of the leaf in
    ///   place (more than one when a leading batch axis is sub-tiled wider than
    ///   1). The product accumulates into the output, so the order over the
    ///   contracted axis is irrelevant.
    pub fn mma(&mut self, lhs: &Tile<'_, E, S>, rhs: &Tile<'_, E, S>) {
        match comptime!(self.kind) {
            TileKind::GmemWhole => {
                // The accumulator is written through its (interior-mutable)
                // views, so a shared reborrow is all the walk needs.
                let out = &*self;

                let space = comptime!(Space::union(&[&out.space, &lhs.space, &rhs.space]));
                let contracted = comptime!(space.contracting(&out.space));
                comptime!(assert!(
                    !contracted.is_empty(),
                    "mma: the output must drop at least one (contracted) axis"
                ));

                let grid = mma_grid::<E, S>(out, lhs, rhs, comptime!(space.clone()));
                let walk = out.partitioner.walk(grid);
                let total = walk.total();
                for i in 0..total {
                    let point = walk.point(i);

                    let a_leaf = lhs.partition(&point);
                    let b_leaf = rhs.partition(&point);
                    let mut acc_leaf = out.partition(&point);

                    acc_leaf.mma(&a_leaf, &b_leaf);
                }
            }
            _ => {
                let out = &*self;
                let matrices = out.matrices();
                for j in 0..matrices {
                    let a = lhs.matrix(j);
                    let b = rhs.matrix(j);
                    let mut acc = out.matrix(j);
                    acc.mma_from(&a, &b);
                }
            }
        }
    }

    /// Copy `src` into `self` — the tile-DSL `copy_from` primitive (a
    /// gmem↔smem element copy, e.g. staging an operand leaf into shared memory),
    /// dispatched on the destination's [`TileKind`].
    ///
    /// A [`GmemWhole`](TileKind::GmemWhole) tensor is not a copy target —
    /// partition it to a leaf first. A leaf
    /// ([`GmemLeaf`](TileKind::GmemLeaf)/[`Smem`](TileKind::Smem)) copies each of
    /// its 2-D [`matrix`](Tile::matrix)s from `src` (more than one when a leading
    /// batch axis is sub-tiled wider than 1).
    pub fn copy_from(&mut self, src: &Tile<'_, E, S>) {
        match comptime!(self.kind) {
            TileKind::GmemWhole => {
                panic!("copy_from: destination must be a leaf or smem tile, not a whole tensor")
            }
            _ => {
                let dst = &*self;
                let matrices = dst.matrices();
                for j in 0..matrices {
                    let mut d = dst.matrix(j);
                    copy_2d::<E, S>(&mut d.view, &src.matrix(j).view);
                }
            }
        }
    }
}

/// The matmul tile [`Grid`] for `space`: each axis's tile count read from an
/// operand that carries it. The partitioner takes the grid from here.
#[cube]
fn mma_grid<E: Numeric, S: Size>(
    out: &Tile<'_, E, S>,
    lhs: &Tile<'_, E, S>,
    rhs: &Tile<'_, E, S>,
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
    out: &Tile<'_, E, S>,
    lhs: &Tile<'_, E, S>,
    rhs: &Tile<'_, E, S>,
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

/// Windows every axis to `[origin, origin + extent)` — the same-rank block
/// [`partition`](Tile::partition) carves out. (The 2-D matrix slice of such a
/// block is [`MatrixWindow`](super::MatrixWindow).)
#[derive(CubeType, Clone)]
pub struct BlockWindow {
    origin: CoordsDyn,
    extent: CoordsDyn,
}

#[cube]
impl BlockWindow {
    pub fn new(origin: CoordsDyn, extent: CoordsDyn) -> Self {
        BlockWindow { origin, extent }
    }
}

#[cube]
impl Layout for BlockWindow {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();
        #[unroll]
        for i in 0..self.origin.len() {
            out.push(self.origin[i] + pos[i]);
        }
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.clone()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;
        #[unroll]
        for i in 0..self.extent.len() {
            valid = valid && pos[i] < self.extent[i];
        }
        valid
    }
}
