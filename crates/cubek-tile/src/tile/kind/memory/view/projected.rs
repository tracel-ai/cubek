//! The gathered read view over a [`Tile`](crate::Tile). [`ProjectionInKernel`] is the [`Layout`] that
//! turns the tile's *logical* coordinate (one per axis of its [`Space`](crate::Space)) into the
//! *physical* coordinate its window is boxed in, applying the operand's [`Projection`].
//!
//! Under the direct mapping the two coincide and this layout is never built; the matmul leaves read
//! through [`TileMatrix`](super::TileMatrix). Under a gathering mapping they differ in rank: a 2-D
//! convolution input has five logical axes over three physical, a step and a tap sharing one.
//!
//! [`CompactionStep`] is the other half, one level down: physical to physical, undoing the lattice a
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
/// window's `CoordsDyn`, cloneable in both worlds so a transparent read addresses the values and
/// the scales through the same one. A blanket impl, bundling bounds rather than naming a concept.
pub(crate) trait LogicalLayout:
    Layout<SourceCoordinates = CoordsDyn> + Clone + 'static + CubeType<ExpandType: Clone>
{
}

impl<L> LogicalLayout for L where
    L: Layout<SourceCoordinates = CoordsDyn> + Clone + 'static + CubeType<ExpandType: Clone>
{
}

/// [`LogicalLayout`] answering a particular coordinate `C`, for the readers that name one.
pub(crate) trait TileLayout<C: Coordinates>:
    LogicalLayout + Layout<Coordinates = C>
{
}

impl<C: Coordinates, L> TileLayout<C> for L where L: LogicalLayout + Layout<Coordinates = C> {}

/// Any [`LogicalLayout`] with an operand's [`Projection`] applied under it: the inner layout
/// resolves a reader's coordinate to the tile's *logical* one, then [`ProjectionInKernel`] folds that
/// onto the window's *physical* one. Under the direct mapping the fold is the identity.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Projected<L: LogicalLayout> {
    inner: L,
    projection: ProjectionInKernel,
}

#[cube]
impl<L: LogicalLayout> Projected<L> {
    pub fn new(inner: L, projection: ProjectionInKernel) -> Self {
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

#[cube]
impl<T: Numeric> Tile<T> {
    /// The whole logical box as a *writable* N-D view: the mutable twin of [`nd`](Tile::nd), for a
    /// caller writing one cell at a time at its logical coordinate. Refused where two logical
    /// positions can share a cell, so a write aliases; a [partition](Composition::Disjoint) cannot.
    pub(crate) fn nd_mut<W: Size>(&mut self) -> MaskedMut<'_, Vector<T, W>, CoordsDyn> {
        let space = comptime!(self.place.space.clone());
        let g = self.mem_mut("nd_mut");
        comptime!(assert!(
            g.projection.composition() != Composition::Overlapping,
            "Tile::nd_mut: an overlapping operand aliases under a write"
        ));
        let layout = g.axis_projection(space);
        g.masked_mut::<W, CoordsDyn, ProjectionInKernel>(layout)
    }

    /// The whole logical box, read through whatever [`Packing`] this tile carries, under the
    /// guard the reader states: the N-D twin of [`matrix_packed`](Tile::matrix_packed), the only
    /// read surface a gathered operand has, and where a procedural tile evaluates its coordinate.
    ///
    /// Under [`Guard::Proved`] the view carries neither the overhang mask nor the window's clamp;
    /// [`guard_provable`](Tile::guard_provable) says when a reader may claim that.
    pub(crate) fn nd_packed<W: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, CoordsDyn> {
        match &self.kind {
            TileKind::Memory(g) => {
                let layout = g.axis_projection(comptime!(self.place.space.clone()));
                g.packed::<W, CoordsDyn, ProjectionInKernel>(layout, guard)
            }
            TileKind::Procedural(data) => {
                procedural_nd::<T, W>(data, comptime!(self.place.space.clone()), guard)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::nd: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::nd: a tma source has no element view"),
            TileKind::Lanes(_) => {
                panic!("Tile::nd: the plane's lanes are read at a coordinate (`scale_at`)")
            }
        }
    }

    /// [`nd_packed`](Tile::nd_packed) at a stated storage element `I` and physical line `WP`.
    pub fn nd<I: Numeric, WP: Size, W: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, CoordsDyn> {
        match &self.kind {
            TileKind::Memory(g) => {
                let layout = g.axis_projection(comptime!(self.place.space.clone()));
                g.transparent::<I, WP, W, CoordsDyn, ProjectionInKernel>(layout, guard)
            }
            TileKind::Procedural(data) => {
                procedural_nd::<T, W>(data, comptime!(self.place.space.clone()), guard)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::nd: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::nd: a tma source has no element view"),
            TileKind::Lanes(_) => {
                panic!("Tile::nd: the plane's lanes are read at a coordinate (`scale_at`)")
            }
        }
    }

    /// The words a packed tile holds, as they lie, one coordinate per axis of the tile's
    /// [`Space`](crate::Space): the twin of [`nd_packed`](Tile::nd_packed) that does not unpack.
    pub(crate) fn nd_words<WP: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<u32, WP>, CoordsDyn> {
        let g = self.mem("nd_words");
        let layout = g.axis_projection(comptime!(self.place.space.clone()));
        g.nd_words::<WP>(layout, guard)
    }

    /// Whether [`Guard::Proved`] would drop a guard no box check can stand in for. A
    /// [`Boundary::Clamp`] axis is the one such guard: a clamped read is in bounds *after*
    /// remapping, so nothing a reader can measure recovers it. The other kinds carry no boundary.
    pub(crate) fn guard_provable(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(data) => {
                comptime!(!data.window.boundaries.contains(&Some(Boundary::Clamp)))
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => comptime!(true),
        }
    }
}

/// A procedural tile's whole box as an N-D view, evaluated at the coordinate. A procedural tile
/// is always scalar-addressed at the leaf, so its direct projection steps by single elements
/// along the innermost axis.
#[cube]
fn procedural_nd<T: Numeric, W: Size>(
    data: &ProceduralData<T>,
    #[comptime] space: Space,
    #[comptime] guard: Guard,
) -> Masked<'_, Vector<T, W>, CoordsDyn> {
    let layout = axis_projection(
        comptime!(space.clone()),
        comptime!(Projection::direct_over(&space)),
        RuntimeMap::integral(comptime!(space.rank())),
        comptime!(1usize),
    );
    Masked::new(
        View::<Vector<T, W>, CoordsDyn>::new::<&ProceduralData<T>, CoordsDyn>(data, layout),
        comptime!(guard.checks() && data.bounds_check),
    )
}

/// A gathered operand split into the map folded once per run and the physical view it addresses.
#[derive(CubeType)]
pub(crate) struct Gathered<'a, T: Numeric, W: Size> {
    pub map: ProjectionInKernel,
    pub view: Masked<'a, Vector<T, W>, CoordsDyn>,
    #[cube(comptime)]
    pub rank: usize,
}

#[cube]
impl<'a, T: Numeric, W: Size> Gathered<'a, T, W> {
    fn new(
        map: ProjectionInKernel,
        view: Masked<'a, Vector<T, W>, CoordsDyn>,
        #[comptime] rank: usize,
    ) -> Self {
        Gathered::<'a, T, W> { map, view, rank }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// The map, physical read surface, and physical rank needed to step a gathered operand by
    /// hand, read through whatever [`Packing`] this tile carries. Constructed together so all
    /// three describe the same memory operand.
    pub(crate) fn nd_split<W: Size>(&self) -> Gathered<'_, T, W> {
        let space = comptime!(self.place.space.clone());
        match &self.kind {
            TileKind::Memory(g) => Gathered::new(
                g.axis_projection(comptime!(space.clone())),
                // A folded caller hands in coordinates it derived itself, so it has proved
                // nothing about them: the window's boundary and the overhang mask both stay on.
                g.packed::<W, CoordsDyn, CompactionStep>(
                    g.physical_box(),
                    comptime!(Guard::Checked),
                ),
                comptime!(g.projection.physical_rank()),
            ),
            TileKind::Procedural(_) | TileKind::Lanes(_) => Gathered::new(
                axis_projection(
                    comptime!(space.clone()),
                    comptime!(Projection::direct_over(&space)),
                    RuntimeMap::integral(comptime!(space.rank())),
                    comptime!(1usize),
                ),
                // The caller steps the map by hand but reads through the tile's own bounds, so
                // it has proved nothing this could drop.
                self.nd_packed::<W>(comptime!(Guard::Checked)),
                comptime!(space.rank()),
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
) -> ProjectionInKernel {
    let rank = comptime!(space.rank());
    let shape = Coords::constant(comptime!(line_extents(&space, vector_size, 0, rank)));

    ProjectionInKernel::new(shape, map, space, projection, vector_size)
}

/// The extents of `space` in `from..to`, the innermost axis converted to a line count by dividing
/// by `vector_size`. Rounded up, matching the buffer these index into: a padded stage's innermost
/// extent need not fill whole lines, and a checked read's box must include the partial last one.
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
