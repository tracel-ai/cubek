//! The 2-D tile world. Everything keyed on `Coords2d` lives here, behind the
//! [`matrix`](super::Tile::matrix) seam — `row`/`col` and the matrix axis pair are
//! confined to this file: the [`matrix`](super::Tile::matrix) bridge that slices an
//! N-D sub-tile into 2-D matrix views, the [`MatrixWindow`] layout, the
//! [`copy_2d`] element copy, and the register `mma`.

use cubecl::{
    prelude::*,
    std::tensor::{
        View, ViewMut,
        layout::{Coords2d, CoordsDyn, Layout, LayoutExpand},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// Slices an N-D [`at`](super::Tile::at) sub-tile to one 2-D matrix:
/// pins the leading (batch) axes to `pins`, exposing the trailing two (matrix)
/// axes. The block origin is already baked into the sub-tile, so this is a pure
/// pin-and-expose with no further offset.
#[derive(CubeType, Clone)]
pub struct MatrixWindow {
    pins: CoordsDyn,
    tile_shape: Coords2d,
}

#[cube]
impl MatrixWindow {
    /// Expose the trailing `rows × cols` matrix under the pinned leading `pins`
    /// (empty for a plain 2-D tile).
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
impl<E: Numeric> Tile<E> {
    /// Number of 2-D matrices in this sub-tile: the product of its leading (batch)
    /// extents (1 for a plain 2-D tile).
    pub fn matrix_count(&self) -> usize {
        let shape = self.view().shape();
        let mut count = u32::from_int(1);
        #[unroll]
        for p in 0..comptime!(self.space.rank() - 2) {
            count *= shape[p];
        }
        count as usize
    }

    /// The [`MatrixWindow`] that slices the `i`-th 2-D matrix out of this sub-tile:
    /// the trailing two axes, with the leading (batch) axes pinned to `i` unraveled
    /// over their extents. Shared by [`matrix`](Tile::matrix) (read) and
    /// [`matrix_mut`](Tile::matrix_mut) (write).
    fn matrix_window(&self, i: usize) -> MatrixWindow {
        let rank = comptime!(self.space.rank());
        let shape = self.view().shape();
        // A sub-tile's matrix shape is its own extents — the partitioner levels are
        // already consumed by the time we slice 2-D matrices.
        let rows = comptime!(self.space.extent(self.space.axis_at(rank - 2)));
        let cols = comptime!(self.space.extent(self.space.axis_at(rank - 1)));

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
        MatrixWindow::new(pins, rows, cols)
    }

    /// The `i`-th 2-D matrix as a read [`View`] (the `lhs`/`rhs` operands).
    pub fn matrix(&self, i: usize) -> View<'_, Vector<E, Const<1>>, Coords2d> {
        let layout = self.matrix_window(i);
        self.view().view(layout)
    }

    /// The `i`-th 2-D matrix as a write [`ViewMut`] (the accumulator / a `stage`
    /// destination).
    pub fn matrix_mut(&mut self, i: usize) -> ViewMut<'_, Vector<E, Const<1>>, Coords2d> {
        let layout = self.matrix_window(i);
        self.view_mut().view_mut(layout)
    }

    /// Accumulate `lhs · rhs` into this tile — the one matmul entry point, at every
    /// level. Dispatched on whether this tile is a [`leaf`](Space::is_leaf): a tile
    /// that still has partitioner levels runs a *lowering* (partition the shared
    /// space, locate each operand at every region, recurse), choosing its move from the
    /// head level's [`Schedule`]; a leaf runs the *contraction*
    /// ([`contract`](Tile::contract)). The lowering's inner `acc.mma(…)` is the
    /// recursion — each `at` consumes one level, so it descends until the
    /// sub-tile is a leaf. A whole-tensor accumulator carries the full stack, so it
    /// always lowers first; a multi-level stack lowers once per level.
    pub fn mma(&mut self, lhs: &Tile<E>, rhs: &Tile<E>) {
        if comptime!(self.space.is_leaf()) {
            self.contract(lhs, rhs);
        } else {
            mma_lower::<E>(lhs, rhs, self);
        }
    }

    /// Locate this tile and both operands at `region`, then [`mma`](Tile::mma) the
    /// sub-tiles — the [`Direct`](Schedule::Direct) lowering's per-region step as one
    /// call. Dispatch (leaf-contract vs. recurse) is `mma`'s, on the *divided* space.
    pub fn mma_at(&mut self, lhs: &Tile<E>, rhs: &Tile<E>, region: &Region) {
        self.at(region).mma(&lhs.at(region), &rhs.at(region));
    }

    /// The leaf: contract `lhs · rhs` into this sub-tile. Reads its own block shape
    /// off the spaces (the accumulator's trailing two axes are the `mr × nr`
    /// matrix; `lhs`'s trailing axis is the contracted `kc`), loops the sub-tile's
    /// 2-D matrices (the batch the partitioner kept in-tile), and runs the register
    /// microkernel on each. The contraction is scalar: operands have no vectorization
    /// type, so `lhs`, `rhs`, and `out` make no assumption about each other's lines.
    pub fn contract(&mut self, lhs: &Tile<E>, rhs: &Tile<E>) {
        // The leaf reads its block shape off the (fully descended) extents: the
        // accumulator's trailing two axes are the `mr × nr` matrix, and `lhs`'s
        // trailing axis is the contracted `kc`.
        let arank = comptime!(self.space.rank());
        let mr = comptime!(self.space.extent(self.space.axis_at(arank - 2)));
        let nr = comptime!(self.space.extent(self.space.axis_at(arank - 1)));
        let kc = comptime!(lhs.space.extent(lhs.space.axis_at(lhs.space.rank() - 1)));

        let matrices = self.matrix_count();
        for j in 0..matrices {
            let l = lhs.matrix(j);
            let r = rhs.matrix(j);
            let mut a = self.matrix_mut(j);
            mma_register::<E>(&l, &r, &mut a, mr, nr, kc);
        }
    }
}

/// One sub-tile's `mr × nr` output block, contracted over `kc`. Every loop
/// unrolls, so the block (`c`), the A column, and the B row all stay in registers:
/// load the block once, run `kc` rank-1 updates ([`outer_product`]), store it back
/// once. No accumulator touches memory in between.
#[cube]
fn mma_register<E: Numeric>(
    lhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    rhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    acc: &mut ViewMut<'_, Vector<E, Const<1>>, Coords2d>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
) {
    let mut c = Array::<Vector<E, Const<1>>>::new(mr * nr);
    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            c[i * nr + j] = acc.read((i as u32, j as u32).runtime());
        }
    }

    #[unroll]
    for p in 0..kc {
        outer_product::<E>(lhs, rhs, &mut c, p, mr, nr);
    }

    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            acc.write((i as u32, j as u32).runtime(), c[i * nr + j]);
        }
    }
}

/// One rank-1 update at depth `p`: load the A column and the B row, accumulate
/// their outer product into the register block `c`. Scalar throughout — operands
/// carry no vectorization type, so the rank-1 update makes no assumption about any
/// operand's line size.
#[cube]
fn outer_product<E: Numeric>(
    lhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    rhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    c: &mut Array<Vector<E, Const<1>>>,
    #[comptime] p: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
) {
    let mut a = Array::<Vector<E, Const<1>>>::new(mr);
    let mut b = Array::<Vector<E, Const<1>>>::new(nr);

    #[unroll]
    for i in 0..mr {
        a[i] = lhs.read((i as u32, p as u32).runtime());
    }
    #[unroll]
    for j in 0..nr {
        b[j] = rhs.read((p as u32, j as u32).runtime());
    }
    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            c[i * nr + j] += a[i] * b[j];
        }
    }
}

/// Element-wise copy of `src` into `dst` (same 2-D shape).
#[cube]
pub fn copy_2d<E: Numeric>(
    dst: &mut ViewMut<'_, Vector<E, Const<1>>, Coords2d>,
    src: &View<'_, Vector<E, Const<1>>, Coords2d>,
) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            dst.write((i, j), src.read((i, j)));
        }
    }
}
