//! A tile cut into slices along one axis, where its holder keeps them ([`AxisSlices`]).

use cubecl::prelude::*;

use crate::*;

/// Who keeps a tile's slices, and so how a slice is read whole.
#[allow(clippy::large_enum_variant)]
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum AxisSlicesKind<E: Float> {
    /// A register block a unit holds whole: a slice runs along its columns, in its registers.
    Registers(RegisterData<E>),
    /// A window of shared memory a plane holds: its units take a slice's cells in turns and meet
    /// on the slice's partials.
    Window(Memory<E>),
    /// A plane's grid of cmma fragments, whose cells lie across its units in the instruction's
    /// own layout: reached only to be scaled, through the plane's scratch.
    Fragments(PlanePartition<E>),
    /// One cmma fragment of a plane's, reached the same way.
    Fragment(PlaneTile<E>),
}

/// A tile cut into slices along `axis`, one for every index of its other axes, each read and
/// written whole where the tile's holder keeps it: the verbs an op reducing or broadcasting along
/// one axis runs. The slices write through to the tile.
///
/// `axis` is the tile's innermost, the one its cells are contiguous along; any other is refused
/// by name.
#[derive(CubeType)]
pub struct AxisSlices<E: Float> {
    kind: AxisSlicesKind<E>,
    #[cube(comptime)]
    space: Space,
    #[cube(comptime)]
    axis: Axis,
}

#[cube]
impl<E: Float> Tile<E> {
    /// This tile cut into slices along `axis`, where its holder keeps them: a unit's register
    /// block, a plane's window of shared memory, or a plane's fragments. A window of memory any
    /// wider holder shares has no one to read a slice whole, and is refused.
    pub fn along(&self, #[comptime] axis: Axis) -> AxisSlices<E> {
        let holder = comptime!(self.place.holder());
        let space = comptime!(self.place.space.clone());
        comptime!(assert!(
            space.axis_at(space.rank() - 1) == axis,
            "Tile::along: a slice runs along the tile's innermost axis, and {axis:?} is not the \
             innermost of {space:?}"
        ));
        let kind = match &self.kind {
            TileKind::PlanePartition(partition) => partition.slices(),
            TileKind::PlaneTile(tile) => tile.slices(),
            TileKind::Memory(window) => {
                comptime!(assert!(
                    holder == ComputeScope::Plane,
                    "Tile::along: a window of shared memory is read slice by slice by the plane \
                     that holds it; this one is held by {holder:?}"
                ));
                AxisSlicesKind::new_Window(window.clone())
            }
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => panic!(
                "Tile::along: a register block, a plane's fragments, or a plane's window holds \
                 slices"
            ),
        };
        AxisSlices::<E> { kind, space, axis }
    }
}

#[cube]
impl<E: Float> AxisSlices<E> {
    /// `self[i, s] = self[i, s] · scale + bias[i, s]`, the procedural `bias` read at each cell's
    /// coordinates, windowed to the same region as the tile.
    pub(crate) fn scale_add(&mut self, scale: E, bias: &Tile<E>) {
        let recipe = bias.recipe();
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => {
                block.scale_add_along(scale, recipe, self.space.clone())
            }
            AxisSlicesKind::Window(window) => {
                window.scale_add_along(scale, recipe, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(_) | AxisSlicesKind::Fragment(_) => {
                AxisSlices::<E>::refuse_reading()
            }
        }
    }

    /// Each slice's max, starting from `seed`'s.
    pub(crate) fn maxima(&self, seed: &Array<E>) -> Array<E> {
        match &self.kind {
            AxisSlicesKind::Registers(block) => block.maxima_along(seed),
            AxisSlicesKind::Window(window) => {
                window.maxima_along(seed, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(_) | AxisSlicesKind::Fragment(_) => {
                AxisSlices::<E>::refuse_reading()
            }
        }
    }

    /// `self[i, s] = exp(self[i, s] − slices[i])` ([`exp_minus_cell`](Self::exp_minus_cell)).
    pub(crate) fn exp_minus(&mut self, slices: &Array<E>) {
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => block.exp_minus_along(slices),
            AxisSlicesKind::Window(window) => {
                window.exp_minus_along(slices, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(_) | AxisSlicesKind::Fragment(_) => {
                AxisSlices::<E>::refuse_reading()
            }
        }
    }

    /// Each slice's sum.
    pub(crate) fn sums(&self) -> Array<E> {
        match &self.kind {
            AxisSlicesKind::Registers(block) => block.sums_along(),
            AxisSlicesKind::Window(window) => window.sums_along(self.space.clone(), self.axis),
            AxisSlicesKind::Fragments(_) | AxisSlicesKind::Fragment(_) => {
                AxisSlices::<E>::refuse_reading()
            }
        }
    }

    /// `self[i, s] *= factors[i]`. A plane's fragments bounce through its scratch, met on
    /// `sync_plane`: every unit of the plane calls it.
    pub fn mul(&mut self, factors: &Array<E>) {
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => block.mul_along(factors, 0usize),
            AxisSlicesKind::Window(window) => {
                window.mul_along(factors, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(partition) => partition.mul_along(factors),
            AxisSlicesKind::Fragment(tile) => tile.mul_along(factors, 0usize),
        }
    }

    /// `exp(value − slice)`, zero where `slice` is itself masked: a slice with nothing live yet,
    /// whose max is still the minimum value.
    pub(crate) fn exp_minus_cell(value: E, slice: E) -> E {
        let live = slice > E::min_value();
        select(live, (value - slice).exp(), E::from_int(0))
    }

    /// A fragment's cells lie across its plane's units in the instruction's own layout, so only
    /// a scale reaches them: to read a slice, drain the fragments into the plane's window first.
    fn refuse_reading() -> ! {
        panic!("AxisSlices: a fragment's slices are read after draining it into its plane's window")
    }
}
