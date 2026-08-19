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
    std::tensor::{
        View,
        layout::{Coordinates, CoordsDyn, Layout, LayoutExpand},
    },
};

use crate::*;

/// The layouts a windowed tile re-views through: any [`Layout`] from a coordinate `C` onto the
/// window's `CoordsDyn`, cloneable in both worlds so the transparent read can address the values
/// and the scales through the same one. A blanket impl, so this bundles bounds rather than naming
/// a new concept; [`BatchMatrix`](super::BatchMatrix) and [`AxisProjection`] are the two the leaves
/// read through, and [`StepUp`] rides the same bounds under a fill.
pub trait LogicalLayout:
    Layout<SourceCoordinates = CoordsDyn> + Clone + 'static + CubeType<ExpandType: Clone>
{
}

impl<L> LogicalLayout for L where
    L: Layout<SourceCoordinates = CoordsDyn> + Clone + 'static + CubeType<ExpandType: Clone>
{
}

/// [`LogicalLayout`] answering a particular coordinate `C`, for the readers that name one.
pub trait TileLayout<C: Coordinates>: LogicalLayout + Layout<Coordinates = C> {}

impl<C: Coordinates, L> TileLayout<C> for L where L: LogicalLayout + Layout<Coordinates = C> {}

/// Any [`LogicalLayout`] with an operand's [`Projection`] applied under it: the inner layout
/// resolves a reader's coordinate to the tile's *logical* one, then [`AxisProjection`] folds that
/// onto the window's *physical* one. Every view goes through this, so the two ranks meet in one
/// place rather than once per reader; under the direct mapping the fold is the identity, which
/// [`Fold`](crate::Fold) collapses to the coordinate itself.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Projected<L: LogicalLayout> {
    inner: L,
    projection: AxisProjection,
}

#[cube]
impl<L: LogicalLayout> Projected<L> {
    pub fn new(inner: L, projection: AxisProjection) -> Self {
        Projected::<L> { inner, projection }
    }
}

#[cube]
impl<L: LogicalLayout> Layout for Projected<L> {
    type Coordinates = L::Coordinates;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        self.projection.to_source_pos(self.inner.to_source_pos(pos))
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let (inner_pos, inner_in_bounds) = self.inner.to_source_pos_checked(pos);
        let (proj_pos, proj_in_bounds) = self.projection.to_source_pos_checked(inner_pos);
        (proj_pos, inner_in_bounds && proj_in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.inner.shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (inner_pos, inner_in_bounds) = self.inner.to_source_pos_checked(pos);
        inner_in_bounds && self.projection.is_in_bounds(inner_pos)
    }
}

/// A [`Layout`] mapping a tile's logical coordinate to its window's physical one:
/// `phys[pa] = (Σ logical[axis] * scale + residue) / divisor`. Sits between the
/// [`Window`](crate::Window) and the element layout: this only folds axes, it never decides what an
/// out-of-range tap does, so the window's own [`Boundary`](crate::Boundary) (zero or the edge cell)
/// still governs the read.
///
/// Static terms a static divisor divides exactly leave the numerator before the floor, which
/// [`PhysicalAxisMap::static_offset_step`](crate::PhysicalAxisMap) decides. A resampling map
/// `⌊(o·scale + r·divisor + residue) / divisor⌋` is read as `⌊(o·scale + residue) / divisor⌋ + r`:
/// the spatial projection stays under the one necessary divide while taps advance by their static
/// physical step. Callers keep spelling the whole affine map, and no projection form spells the
/// split.
///
/// Constant offsets are handled by [`Window`](crate::Window) and omitted here, all but the part a
/// window origin cannot absorb: under a division the offset sets the phase the floor starts at, and
/// that phase, which the [`RuntimeMap`](crate::RuntimeMap) carries, has to be inside the numerator.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct AxisProjection {
    /// The tile's per-logical-axis extents, in the space's axis order. The innermost is a
    /// line count, matching the window's innermost physical axis.
    shape: Coords<u32>,
    /// The projection's runtime half: the coefficients a tap folds through, and the phase its
    /// window origin left over. The constant offsets are not among them: a tap is relative to the
    /// window, which they placed.
    map: RuntimeMap,
    #[cube(comptime)]
    space: Space,
    #[cube(comptime)]
    projection: Projection,
}

#[cube]
impl AxisProjection {
    pub fn new(
        shape: Coords<u32>,
        map: RuntimeMap,
        #[comptime] space: Space,
        #[comptime] projection: Projection,
    ) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == space.rank(),
            "AxisProjection: shape has {rank} entries but the space spans {} logical axes",
            space.rank()
        ));
        let given = map.coefficients.len();
        comptime!(assert!(
            given == projection.dynamic_coefficient_count(),
            "AxisProjection: the projection has {} Dynamic coefficients and divisors but {given} \
             were given",
            projection.dynamic_coefficient_count()
        ));
        let phases = map.residues.len();
        comptime!(assert!(
            phases == projection.physical_rank(),
            "AxisProjection: the projection has {} physical axes but {phases} residues were given",
            projection.physical_rank()
        ));
        AxisProjection {
            shape,
            map,
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
            let axis_map = comptime!(self.projection.physical_axis(pa));
            let n = comptime!(axis_map.terms().len());

            // Per-term products left in the numerator, summed below (chained, so a single
            // coefficient-1 term folds to the coordinate itself). Under a division the sum starts
            // at the phase the window origin could not absorb.
            let mut terms = Coords::<u32>::new();
            if comptime!(axis_map.is_rational()) {
                terms.push(self.map.residues.at(pa));
            }
            // The exact terms, held apart so the numerator above is the same expression for every
            // tap: a gather then computes one spatial floor and adds a step to it, where a single
            // sum would put every tap's coordinate under the divide and defeat the reuse.
            let mut offsets = Coords::<u32>::new();
            #[unroll]
            for t in 0..n {
                let term = comptime!(axis_map.terms()[t]);
                let p = comptime!(self.space.position(term.axis));
                match comptime!(axis_map.static_offset_step(t)) {
                    Some(step) => offsets.push(pos[p].fmul(comptime!(step as u32))),
                    None => match comptime!(term.scale) {
                        Scale::Static(s) => terms.push(pos[p].fmul(comptime!(s as u32))),
                        Scale::Dynamic { .. } => terms.push(pos[p].fmul(self.map.coefficients.at(
                            comptime!(self.projection.dynamic_scale_index(pa, t).unwrap()),
                        ))),
                    },
                }
            }
            let n_kept = terms.len();
            let n_exact = offsets.len();
            let sum = terms.fsum(comptime!((0..n_kept).collect::<Vec<_>>()));

            if comptime!(axis_map.is_rational()) {
                match comptime!(axis_map.divisor()) {
                    // No offsets when the divisor is dynamic, and `fadd` folds the empty sum away,
                    // so only the static arm spells the addition.
                    Divisor::Static(d) => {
                        let offset = offsets.fsum(comptime!((0..n_exact).collect::<Vec<_>>()));
                        out.push(sum.fdiv(comptime!(d as u32)).fadd(offset));
                    }
                    Divisor::Dynamic { .. } => {
                        let divisor = self.map.coefficients.at(comptime!(
                            self.projection.dynamic_divisor_index(pa).unwrap()
                        ));
                        out.push(sum.fdiv(divisor));
                    }
                }
            } else {
                out.push(sum);
            }
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
        within(&self.shape, pos)
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

    /// The compacted box, the source cell it steps up to being the [`Window`](crate::Window)'s
    /// question.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        within(&self.shape, pos)
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
                    g.map.clone(),
                    self.vector_size(),
                );
                g.nd_transparent::<I, WP, W>(layout)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::nd: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::nd: a tma source has no element view"),
            TileKind::Procedural(data) => {
                let layout = axis_projection(
                    comptime!(self.space.clone()),
                    comptime!(Projection::direct_over(&self.space)),
                    RuntimeMap::integral(comptime!(self.space.rank())),
                    comptime!(1usize),
                );
                MaskedView::new(
                    View::<Vector<T, W>, CoordsDyn>::new::<&ProceduralData<T>, CoordsDyn>(
                        data, layout,
                    ),
                    comptime!(data.bounds_check),
                )
            }
        }
    }
}

/// The tile's per-axis extents paired with its operand's mapping. `Space` is scalar; the
/// innermost axis is a line count, matching the window it indexes into.
#[cube]
pub(crate) fn axis_projection(
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    map: RuntimeMap,
    #[comptime] vector_size: usize,
) -> AxisProjection {
    let rank = comptime!(space.rank());
    let shape = const_coords(comptime!(line_extents(&space, vector_size, 0, rank)));

    AxisProjection::new(shape, map, space, projection)
}

/// Returns the extents of `space` in the range `from..to`, with the innermost axis
/// converted to line count by dividing by `vector_size`.
pub(crate) fn line_extents(
    space: &Space,
    vector_size: usize,
    from: usize,
    to: usize,
) -> Vec<usize> {
    let last = space.rank() - 1;
    (from..to)
        .map(|p| {
            let e = space.extent_at(p);
            if p == last { e / vector_size } else { e }
        })
        .collect()
}
