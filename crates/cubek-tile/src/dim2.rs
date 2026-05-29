//! The 2-D leaf world. Everything keyed on `Coords2d` lives here, behind the
//! [`matrix`](super::Tile::matrix) seam — `row`/`col` and the matrix axis pair
//! are confined to this file. Holds the [`Tile2d`] alias, the
//! [`matrix`](super::Tile::matrix)/[`matrices`](super::Tile::matrices) bridge
//! that slices an N-D leaf into 2-D matrices, the [`MatrixWindow`] leaf layout,
//! the smem staging buffer ([`SmemLayout`]/[`stage_smem`]), and the 2-D element
//! mechanisms ([`copy_2d`], [`mma_2d`]) the [`Tile`](super::Tile) pillars
//! delegate to.

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// A tile collapsed to a fixed 2-D leaf: what [`partition`](super::Tile::partition)
/// yields and what the 2-D ops ([`copy_from`](Tile::copy_from) and the smem
/// staging) speak. The collapse is one-way — a `Tile2d` has no `partition`, so
/// climbing back to a higher-rank space would take an explicit extend/broadcast.
pub type Tile2d<'a, E, S> = TileOf<'a, E, S, Coords2d>;

/// Slices an N-D [`partition`](super::Tile::partition) leaf to one 2-D matrix:
/// pins the leading (batch) axes to `pins`, exposing the trailing two (matrix)
/// axes. The block origin is already baked into the leaf, so this is a pure
/// pin-and-expose with no further offset.
#[derive(CubeType, Clone)]
pub struct MatrixWindow {
    pins: CoordsDyn,
    tile_shape: Coords2d,
}

#[cube]
impl MatrixWindow {
    /// Expose the trailing `rows × cols` matrix under the pinned leading `pins`
    /// (empty for a plain 2-D leaf).
    pub fn new(pins: CoordsDyn, #[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        MatrixWindow {
            pins,
            tile_shape: (
                u32::from_int(comptime!(rows as i64)),
                u32::from_int(comptime!(cols as i64)),
            ),
        }
    }
}

#[cube]
impl Layout for MatrixWindow {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (t0, t1) = pos;
        let mut out = self.pins.clone();
        out.push(t0);
        out.push(t1);
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos);
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.tile_shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (t0, t1) = pos;
        let (s0, s1) = self.tile_shape;
        t0 < s0 && t1 < s1
    }
}

#[cube]
impl<'a, E: Numeric, S: Size> TileOf<'a, E, S, CoordsDyn> {
    /// Number of 2-D matrices in this leaf: the product of its leading (batch)
    /// extents (1 for a plain 2-D leaf).
    pub fn matrices(&self) -> usize {
        let shape = self.view.shape();
        let mut count = u32::from_int(1);
        #[unroll]
        for p in 0..comptime!(self.space.rank() - 2) {
            count *= shape[p];
        }
        count as usize
    }

    /// The `i`-th 2-D matrix: the trailing two axes exposed as a [`Tile2d`], with
    /// the leading (batch) axes pinned to `i` unraveled over their extents.
    pub fn matrix(&self, i: usize) -> Tile2d<'a, E, S> {
        let rank = comptime!(self.space.rank());
        let shape = self.view.shape();
        let rows = self
            .partitioner
            .sub_tile_edge(comptime!(self.space.axis_at(rank - 2)));
        let cols = self
            .partitioner
            .sub_tile_edge(comptime!(self.space.axis_at(rank - 1)));

        // Unravel `i` (row-major) over the leading extents into pinned coords.
        let mut pins = CoordsDyn::new();
        #[unroll]
        for p in 0..comptime!(rank - 2) {
            let mut weight = u32::from_int(1);
            #[unroll]
            for q in comptime!(p + 1)..comptime!(rank - 2) {
                weight *= shape[q];
            }
            pins.push((i as u32 / weight) % shape[p]);
        }

        let layout = MatrixWindow::new(pins, rows, cols);
        TileOf::<'a, E, S, Coords2d> {
            view: self.view.clone().view_mut(layout),
            partitioner: self.partitioner.clone(),
            space: comptime!(self.space.clone()),
            kind: comptime!(TileKind::GmemLeaf),
        }
    }
}

/// Row-major layout over a flat smem buffer holding a whole N-D leaf
/// (`[d0, …, rows, cols]`), so a staged leaf is addressed exactly like the
/// global leaf it mirrors — and [`matrix`](super::Tile::matrix) slices it the
/// same way.
#[derive(CubeType, Clone)]
pub struct SmemLayout {
    shape: CoordsDyn,
    strides: CoordsDyn,
}

#[cube]
impl Layout for SmemLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut idx = u32::from_int(0);
        #[unroll]
        for i in 0..self.strides.len() {
            idx += pos[i] * self.strides[i];
        }
        idx as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;
        #[unroll]
        for i in 0..self.shape.len() {
            valid = valid && pos[i] < self.shape[i];
        }
        valid
    }
}

/// The row-major layout for a leaf of `space` at `partitioner`'s sub-tile edges —
/// the smem twin of a [`partition`](super::Tile::partition) leaf.
#[cube]
pub fn smem_layout(#[comptime] space: Space, partitioner: &Partitioner) -> SmemLayout {
    let rank = comptime!(space.rank());

    let mut shape = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        let edge = partitioner.sub_tile_edge(comptime!(space.axis_at(p)));
        shape.push(u32::from_int(comptime!(edge as i64)));
    }

    // Row-major strides: stride[p] = product of the trailing extents.
    let mut strides = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        let mut weight = u32::from_int(1);
        #[unroll]
        for q in comptime!(p + 1)..rank {
            weight *= shape[q];
        }
        strides.push(weight);
    }

    SmemLayout { shape, strides }
}

#[cube]
impl<'a, E: Numeric, S: Size> TileOf<'a, E, S, Coords2d> {
    /// Accumulate `lhs · rhs` into this matrix — the 2-D leaf of the tile-DSL
    /// [`mma`](Tile::mma) primitive.
    pub fn mma_from(&mut self, lhs: &Tile2d<'_, E, S>, rhs: &Tile2d<'_, E, S>) {
        mma_2d::<E, S>(&mut self.view, &lhs.view, &rhs.view);
    }
}

/// Wrap a shared-memory view as a whole-leaf [`Smem`](TileKind::Smem) tile — the
/// staging destination for [`copy_from`](super::Tile::copy_from).
#[cube]
pub fn stage_smem<'a, E: Numeric, S: Size>(
    view: ViewMut<'a, Vector<E, S>, CoordsDyn>,
    #[comptime] space: Space,
    partitioner: Partitioner,
) -> Tile<'a, E, S> {
    TileOf::<'a, E, S, CoordsDyn> {
        view,
        partitioner,
        space,
        kind: comptime!(TileKind::Smem),
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

/// Scalar 2-D contraction `acc(i, j) += Σ_c lhs(i, c) · rhs(c, j)`; shapes read
/// from the views.
#[cube]
pub fn mma_2d<E: Numeric, S: Size>(
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
