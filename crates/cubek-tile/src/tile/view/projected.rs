//! The gathered read view over a [`Tile`](crate::Tile). [`AxisProjection`] is the [`Layout`] that
//! turns the tile's *logical* coordinate (one per axis of its [`Space`](crate::Space)) into the
//! *physical* coordinate its window is boxed in, applying the operand's [`Projection`].
//!
//! Under the direct mapping the two coincide and this layout is never built; the matmul leaves keep
//! reading through [`BatchMatrix`](super::BatchMatrix). Under a gathering mapping they differ in
//! rank: a 2-D convolution input carries five logical axes over three physical ones, and two
//! logical coordinates (an output step and a tap) address the same physical axis.
//!
//! [`StepUp`] is the other half, one level down: physical to physical, undoing the lattice a
//! [`Compaction`] quotients a gathered window by, so a fill of the compacted stage lands on the
//! source cells the stage keeps.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coordinates, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// The layouts a windowed tile re-views through: any [`Layout`] from a coordinate `C` onto the
/// window's `CoordsDyn`, cloneable in both worlds so the transparent read can address the values
/// and the scales through the same one. A blanket impl, so this bundles bounds rather than naming
/// a new concept; [`BatchMatrix`](super::BatchMatrix) and [`AxisProjection`] are the two the leaves
/// read through, and [`StepUp`] rides the same bounds under a fill.
pub trait TileLayout<C: Coordinates>:
    Layout<Coordinates = C, SourceCoordinates = CoordsDyn>
    + Clone
    + 'static
    + CubeType<ExpandType: Clone>
{
}

impl<C: Coordinates, L> TileLayout<C> for L where
    L: Layout<Coordinates = C, SourceCoordinates = CoordsDyn>
        + Clone
        + 'static
        + CubeType<ExpandType: Clone>
{
}

/// A [`Layout`] mapping a tile's logical coordinate to its window's physical one:
/// `phys[pa] = Σ logical[axis] * scale`. Sits between the [`Window`](crate::Window) and the
/// element layout, so the window's `bound` still masks the read (an out-of-range stencil tap
/// reads zero).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct AxisProjection {
    /// The tile's per-logical-axis extents, in the space's axis order. The innermost is a
    /// line count, matching the window's innermost physical axis.
    shape: Coords<u32>,
    #[cube(comptime)]
    space: Space,
    #[cube(comptime)]
    projection: Projection,
}

#[cube]
impl AxisProjection {
    pub fn new(
        shape: Coords<u32>,
        #[comptime] space: Space,
        #[comptime] projection: Projection,
    ) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == space.rank(),
            "AxisProjection: shape has {rank} entries but the space spans {} logical axes",
            space.rank()
        ));
        AxisProjection {
            shape,
            space,
            projection,
        }
    }
}

#[cube]
impl Layout for AxisProjection {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            let n = comptime!(self.projection.physical_axis(pa).terms().len());
            // Per-term products, summed below (chained, so a single coefficient-1 term folds to
            // the coordinate itself).
            let mut terms = Coords::<u32>::new();
            #[unroll]
            for t in 0..n {
                let term = comptime!(self.projection.physical_axis(pa).terms()[t]);
                let p = comptime!(self.space.position(term.axis));
                terms.push(pos[p].fmul(comptime!(term.scale.get() as u32)));
            }
            out.push(terms.fsum(comptime!((0..n).collect::<Vec<_>>())));
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.to_dyn()
    }

    /// The logical box. Whether the physical coordinate it maps to is within the operand's valid
    /// data is the [`Window`](crate::Window)'s question, asked one layer down.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;

        #[unroll]
        for p in 0..self.shape.len() {
            valid = valid && pos[p] < self.shape.at(p);
        }

        valid
    }
}

/// A [`Layout`] scaling a *physical* coordinate by one step per axis: `src[pa] = pos[pa] * step`.
/// The inverse of the lattice a [`Compaction`] quotients a gathered operand's window by, so a fill
/// walking the compacted stage lands on the source cells the stage keeps.
///
/// Only built when the compaction has a step to undo; a dense window (every direct operand, and any
/// gather whose taps are adjacent) is read without this layer at all.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct StepUp {
    /// The compacted extents this steps through, innermost a line count.
    shape: Coords<u32>,
    #[cube(comptime)]
    steps: Vec<usize>,
}

#[cube]
impl StepUp {
    pub fn new(shape: Coords<u32>, #[comptime] steps: Vec<usize>) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == steps.len(),
            "StepUp: shape has {rank} entries but {} steps were given",
            steps.len()
        ));
        StepUp { shape, steps }
    }
}

#[cube]
impl Layout for StepUp {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.steps.len()) {
            out.push(pos[pa].fmul(comptime!(self.steps[pa] as u32)));
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.to_dyn()
    }

    /// The compacted box. Whether the source cell it steps up to holds valid data is the
    /// [`Window`](crate::Window)'s question, asked one layer down.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;

        #[unroll]
        for p in 0..self.shape.len() {
            valid = valid && pos[p] < self.shape.at(p);
        }

        valid
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// This tile's whole logical box as a quantization-transparent read view, one coordinate per
    /// axis of its [`Space`](crate::Space) (the innermost a line index). The N-D counterpart of
    /// [`matrix_transparent`](Tile::matrix_transparent), and the only read surface a gathered
    /// operand has: its logical rank exceeds its buffer's, so no 2-D window describes it.
    pub fn nd<I: Numeric, WP: Size, W: Size>(&self) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let layout = axis_projection(
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    self.vector_size(),
                );
                g.nd_transparent::<I, WP, W>(layout)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::nd: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::nd: a tma source has no element view"),
        }
    }
}

/// The tile's per-axis extents paired with its operand's mapping. `Space` is scalar; the
/// innermost axis is a line count, matching the window it indexes into.
#[cube]
pub(crate) fn axis_projection(
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    #[comptime] vector_size: usize,
) -> AxisProjection {
    let rank = comptime!(space.rank());
    let last = comptime!(rank - 1);
    let mut shape = Coords::<u32>::new();

    #[unroll]
    for p in 0..rank {
        let e = comptime!(space.extent_at(p));
        shape.push(comptime!((if p == last { e / vector_size } else { e }) as u32).runtime());
    }

    AxisProjection::new(shape, space, projection)
}
