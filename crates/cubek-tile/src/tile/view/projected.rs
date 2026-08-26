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

/// The static physical step a term contributes once it is taken out of its axis's evaluation:
/// under a floor only what the divisor factors out
/// ([`static_offset_step`](PhysicalAxisMap::static_offset_step)), elsewhere the coefficient itself.
/// `None` for a dynamic coefficient outside a floor, which the kernel reads at runtime instead.
///
/// Panics for a term a rational axis keeps inside its floor: its contribution to the physical
/// coordinate is not additive there, so no walk can step past it and the map has to be folded at
/// every position.
fn split_step(map: &PhysicalAxisMap, term: usize) -> Option<usize> {
    if map.is_rational() {
        return Some(map.static_offset_step(term).unwrap_or_else(|| {
            panic!(
                "AxisProjection::advance: {:?} stays inside this axis's floor, so stepping it is \
                 not an addition and the map has to be folded at every position",
                map.terms()[term].axis
            )
        }));
    }
    match map.terms()[term].scale {
        Scale::Static(s) => Some(s),
        Scale::Dynamic { .. } => None,
    }
}

#[cube]
impl AxisProjection {
    /// The source coordinate of `pos` with every axis in `moving` held at zero: the part of the
    /// map a walk over those axes leaves alone, which [`advance`](Self::advance) puts back.
    ///
    /// The rational axes are what the split buys. Their numerator is the same expression at every
    /// point of such a walk, so a gather takes one floor per accumulator cell where folding the
    /// whole map takes one per tap.
    pub fn anchor(&self, pos: CoordsDyn, #[comptime] moving: Vec<Axis>) -> CoordsDyn {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            out.push(self.project_axis(&pos, pa, comptime!(moving.clone())));
        }

        out
    }

    /// [`anchor`](Self::anchor) for one physical axis. Factor-local boundary normalization uses
    /// this narrow form so checking one tap does not rebuild unrelated source coordinates.
    pub(crate) fn project_axis(
        &self,
        pos: &CoordsDyn,
        #[comptime] pa: usize,
        #[comptime] moving: Vec<Axis>,
    ) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa));
        let n = comptime!(axis_map.terms().len());

        // Per-term products left in the numerator, summed below (chained, so a single
        // coefficient-1 term folds to the coordinate itself). Under a division the sum starts at
        // the phase the window origin could not absorb.
        let mut terms = Coords::<u32>::new();
        if comptime!(axis_map.is_rational()) {
            terms.push(self.map.residues.at(pa));
        }
        // Exact steps stay outside the numerator so a rational projection takes one spatial floor
        // and adds the tap step to it.
        let mut offsets = Coords::<u32>::new();
        #[unroll]
        for t in 0..n {
            let term = comptime!(axis_map.terms()[t]);
            if comptime!(!moving.contains(&term.axis)) {
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
        }
        let n_kept = terms.len();
        let n_exact = offsets.len();
        let sum = terms.fsum(comptime!((0..n_kept).collect::<Vec<_>>()));

        if comptime!(axis_map.is_rational()) {
            match comptime!(axis_map.divisor()) {
                Divisor::Static(d) => {
                    let offset = offsets.fsum(comptime!((0..n_exact).collect::<Vec<_>>()));
                    sum.fdiv(comptime!(d as u32)).fadd(offset)
                }
                Divisor::Dynamic { .. } => {
                    let divisor = self.map.coefficients.at(comptime!(
                        self.projection.dynamic_divisor_index(pa).unwrap()
                    ));
                    sum.fdiv(divisor)
                }
            }
        } else {
            sum
        }
    }

    /// `anchor` moved to where `pos` places the `moving` axes, which must be the ones it was
    /// [anchored](Self::anchor) against.
    ///
    /// Every one of them enters linearly, so the move is an exact addition: outside a division by
    /// the term's own coefficient, and under one by the static step the divisor factors out of the
    /// floor.
    pub fn advance(
        &self,
        anchor: &CoordsDyn,
        pos: CoordsDyn,
        #[comptime] moving: Vec<Axis>,
    ) -> CoordsDyn {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            let axis_map = comptime!(self.projection.physical_axis(pa));
            let n = comptime!(axis_map.terms().len());

            let mut steps = Coords::<u32>::new();
            steps.push(anchor[pa]);
            #[unroll]
            for t in 0..n {
                let term = comptime!(axis_map.terms()[t]);
                if comptime!(moving.contains(&term.axis)) {
                    let p = comptime!(self.space.position(term.axis));
                    match comptime!(split_step(axis_map, t)) {
                        Some(step) => steps.push(pos[p].fmul(comptime!(step as u32))),
                        None => steps.push(pos[p].fmul(self.map.coefficients.at(comptime!(
                            self.projection.dynamic_scale_index(pa, t).unwrap()
                        )))),
                    }
                }
            }
            let n_steps = steps.len();
            out.push(steps.fsum(comptime!((0..n_steps).collect::<Vec<_>>())));
        }

        out
    }
}

#[cube]
impl Layout for AxisProjection {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        self.anchor(pos, comptime!(Vec::new()))
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
    /// The whole logical box, read through whatever [`Packing`] this tile carries, under the
    /// guard the reader states. The N-D twin of [`matrix_packed`](Tile::matrix_packed).
    pub fn nd_packed<W: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        let served = self.vector_size();
        let packing = self.packing();
        let physical = comptime!(packing.physical(served));
        match comptime!(packing) {
            Packing::Plain => {
                let size!(WP) = physical;
                self.nd::<T, WP, W>(guard)
            }
            Packing::Native => {
                let size!(WP) = physical;
                self.nd::<i8, WP, W>(guard)
            }
            Packing::Packed { factor: _ } => {
                let size!(WP) = physical;
                self.nd::<u32, WP, W>(guard)
            }
        }
    }

    /// Whether [`Guard::Proved`] would drop a guard no box check can stand in for.
    ///
    /// Kept next to [`nd`](Tile::nd), the operation that drops it. A [`Boundary::Clamp`] axis is
    /// the one such guard: a clamped read is in bounds *after* remapping, so the window reports
    /// it in bounds whatever its raw coordinate was, and nothing a reader can measure recovers
    /// that coordinate. The remaining kinds carry no [`Window`](crate::Window) boundary at all;
    /// the ones that cannot form an N-D view are rejected by `nd` itself.
    pub(crate) fn guard_provable(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Gmem(data) | TileKind::Smem(data) => {
                comptime!(!data.window.boundaries.contains(&Some(Boundary::Clamp)))
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => comptime!(true),
        }
    }

    /// [`nd_packed`](Tile::nd_packed) at a stated storage element, one coordinate per axis of the
    /// tile's [`Space`](crate::Space) (the innermost a line index). The only read surface a
    /// gathered operand has: its logical rank exceeds its buffer's, so no 2-D window describes it.
    ///
    /// Under [`Guard::Proved`] the view carries neither the overhang mask nor the window's clamp,
    /// which are what a checked leaf pays per access; [`guard_provable`](Tile::guard_provable)
    /// says when that is a claim a reader is allowed to make.
    pub fn nd<I: Numeric, WP: Size, W: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                let layout = axis_projection(
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    self.vector_size(),
                );
                g.nd_transparent::<I, WP, W>(layout, guard)
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
                    comptime!(guard.checks() && data.bounds_check),
                )
            }
        }
    }
}

/// A gathered operand split into the map folded once per run and the physical view it addresses.
#[derive(CubeType)]
pub struct NdReader<'a, T: Numeric, W: Size> {
    pub map: AxisProjection,
    pub view: MaskedView<'a, Vector<T, W>, CoordsDyn>,
    #[cube(comptime)]
    pub rank: usize,
}

#[cube]
impl<'a, T: Numeric, W: Size> NdReader<'a, T, W> {
    fn new(
        map: AxisProjection,
        view: MaskedView<'a, Vector<T, W>, CoordsDyn>,
        #[comptime] rank: usize,
    ) -> Self {
        NdReader::<'a, T, W> { map, view, rank }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// [`nd_split`](Tile::nd_split) with this tile's quant packing resolved.
    pub fn nd_split_packed<W: Size>(&self) -> NdReader<'_, T, W> {
        let served = self.vector_size();
        let packing = self.packing();
        let physical = comptime!(packing.physical(served));
        match comptime!(packing) {
            Packing::Plain => {
                let size!(WP) = physical;
                self.nd_split::<T, WP, W>()
            }
            Packing::Native => {
                let size!(WP) = physical;
                self.nd_split::<i8, WP, W>()
            }
            Packing::Packed { factor: _ } => {
                let size!(WP) = physical;
                self.nd_split::<u32, WP, W>()
            }
        }
    }

    /// The map, physical read surface, and physical rank needed to step a gathered operand by
    /// hand. Constructed together so all three describe the same memory operand.
    pub fn nd_split<I: Numeric, WP: Size, W: Size>(&self) -> NdReader<'_, T, W> {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => NdReader::new(
                axis_projection(
                    comptime!(self.space.clone()),
                    comptime!(g.projection.clone()),
                    g.map.clone(),
                    self.vector_size(),
                ),
                g.nd_physical::<I, WP, W>(),
                comptime!(g.projection.physical_rank()),
            ),
            // A procedural tile is always scalar-addressed at the leaf (`vector_size() == 1`,
            // enforced by `ProceduralDataExpand::__expand_vector_size_method`). The direct
            // projection therefore steps by single elements along the innermost axis.
            TileKind::Procedural(_) => NdReader::new(
                axis_projection(
                    comptime!(self.space.clone()),
                    comptime!(Projection::direct_over(&self.space)),
                    RuntimeMap::integral(comptime!(self.space.rank())),
                    comptime!(1usize),
                ),
                // The caller steps the map by hand but reads through the tile's own bounds, so
                // it has proved nothing this could drop.
                self.nd::<I, WP, W>(comptime!(Guard::Checked)),
                comptime!(self.space.rank()),
            ),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                panic!("Tile::nd_split: this tile has no addressable N-D read surface")
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
///
/// Rounded up, matching the buffer these extents index into (`storage_extents` and
/// `Compaction::line_extents`): a padded stage's innermost extent need not fill whole lines, and
/// the box a read is bounds-checked against has to include the partial last one the stage really
/// holds.
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
            if p == last {
                e.div_ceil(vector_size)
            } else {
                e
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUT: Axis = Axis(0);
    const TAP: Axis = Axis(1);

    /// Outside a floor a term steps by its own coefficient, which is what makes `advance` an
    /// addition.
    #[test]
    fn a_plain_affine_term_steps_by_its_coefficient() {
        let map = PhysicalAxisMap::affine(&[(OUT, 2), (TAP, 3)]);
        assert_eq!(split_step(&map, 0), Some(2));
        assert_eq!(split_step(&map, 1), Some(3));
    }

    /// A dynamic coefficient is a runtime read, not a static step.
    #[test]
    fn a_dynamic_coefficient_has_no_static_step() {
        let map =
            PhysicalAxisMap::scaled(&[(OUT, Scale::Static(2)), (TAP, Scale::Dynamic { max: 4 })]);
        assert_eq!(split_step(&map, 1), None);
    }

    /// Under a floor a term steps by what the divisor factors out: `⌊(x + m·4)/2⌋` moves by `2`
    /// per `m`, exactly.
    #[test]
    fn a_divisible_term_steps_by_what_the_floor_factors_out() {
        let map = PhysicalAxisMap::affine(&[(OUT, 3), (TAP, 4)]).over(2);
        assert_eq!(split_step(&map, 1), Some(2));
    }

    /// The same map's indivisible term: `⌊(3·out + …)/2⌋` is not `out` times anything, so it has
    /// to stay anchored rather than be stepped.
    #[test]
    #[should_panic(expected = "stays inside this axis's floor")]
    fn an_indivisible_term_under_a_floor_cannot_be_stepped() {
        let map = PhysicalAxisMap::affine(&[(OUT, 3), (TAP, 4)]).over(2);
        split_step(&map, 0);
    }
}
