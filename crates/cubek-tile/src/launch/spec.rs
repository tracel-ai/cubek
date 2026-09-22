//! The comptime half of an operand: which axes its buffer spans and how they address it.

use cubecl::zspace::SmallVec;

use crate::*;

/// The comptime half of an operand: which axes of the kernel's one [`Space`] its buffer spans and
/// how they address its physical axes ([`Projection`], the storage tiling in its own repetition).
/// [`Tile::of`](crate::Tile::of) projects that space onto them, so no operand carries a copy.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileSpec {
    /// How this operand's logical axes address its buffer's physical ones.
    pub projection: Projection,
    /// How each coordinate axis handles out-of-bounds access. `None` is the unchecked fast path;
    /// an empty list means every axis is unchecked.
    pub boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    /// The launch's cube size (units per cube), `0` when unknown; carried into every stage
    /// filled from this operand, which emits its fill straight-line when it knows the count.
    pub units: usize,
    /// How this operand's values sit in its binding ([`Packing::Plain`] unless stated): the one
    /// fact a packed operand needs and a plain binding cannot carry, since a `u32` says nothing
    /// of the values in it. Stated by [`packed`](Self::packed) or a quantized operand's scheme.
    pub packing: Packing,
    /// What this operand's storage tiles are to the windows it is read through. Settled by
    /// the launch; [`Strided`](Storage::Strided) for every untiled operand, which is every operand
    /// that does not say otherwise.
    pub storage: Storage,
}

impl TileSpec {
    /// An operand's spec from its mapping; the optional halves take safe defaults (unchecked, cube
    /// size unknown) and have setters. [`Projection::validate`] does not run here: its rule on the
    /// innermost identity needs the served width, which only [`Tile::of`](crate::Tile::of) knows.
    pub fn new(projection: Projection) -> Self {
        TileSpec {
            projection,
            boundaries: SmallVec::new(),
            units: 0,
            packing: Packing::Plain,
            storage: Storage::Strided,
        }
    }

    /// [`new`](Self::new) over the [`direct`](Projection::direct) mapping: one logical axis per
    /// physical axis, which is every non-gather, untiled operand.
    pub fn direct(axes: &[Axis]) -> Self {
        Self::new(Projection::direct(axes))
    }

    /// The axes this operand spans, in its logical order.
    pub fn axes(&self) -> &[Axis] {
        self.projection.logical_axes()
    }

    /// State that this operand's binding holds `u32` words packing several values each, one per
    /// `field`-wide slot, innermost axis first; the tile unpacks at the read. Values only: scales
    /// are a second tensor, folded in by a verb the kernel writes, so a q4 kernel needs no scheme.
    pub fn packed(self, field: impl Into<Field>) -> Self {
        self.packing(Packing::Packed {
            field: field.into(),
        })
    }

    /// [`packed`](Self::packed) for a caller holding the [`Packing`] itself, which the launch
    /// derivation does.
    pub fn packing(mut self, packing: Packing) -> Self {
        self.packing = packing;
        self
    }

    /// What this operand's storage tiles are to the windows it is read through; settled by
    /// the launch, [`Strided`](Storage::Strided) by default (every untiled operand).
    pub fn storage(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }

    /// Set whether edge reads/writes must be bounds-checked with [`Boundary::Zero`]. Overwrites
    /// whatever mode stood, so sequence a `Clamp` [`with_boundary`](Self::with_boundary) *after*
    /// this, not before.
    pub fn checked(self, check: bool) -> Self {
        self.with_boundary(check.then_some(Boundary::Zero))
    }

    /// Set one boundary handling mode for out-of-bounds access on every coordinate axis. Flattens
    /// whatever per-axis list [`boundaries`](Self::boundaries) may have set, so sequence the
    /// per-axis call after this one, not before it.
    pub fn with_boundary(mut self, boundary: Option<Boundary>) -> Self {
        let coord_rank = self.projection.coordinate_rank();
        self.boundaries = match boundary {
            Some(b) => SmallVec::from_elem(Some(b), coord_rank),
            None => SmallVec::new(),
        };
        self
    }

    /// State the boundary mode for every coordinate axis; an all-`None` list collapses to the empty
    /// one, so "nothing is checked" has one representation. Shaped over the
    /// [`coordinate_rank`](Projection::coordinate_rank), not the storage-tiled physical rank.
    pub fn boundaries(mut self, boundaries: &[Option<Boundary>]) -> Self {
        let coord_rank = self.projection.coordinate_rank();
        assert!(
            boundaries.len() == coord_rank,
            "TileSpec::boundaries: {} modes stated but the operand has {coord_rank} coordinate \
             axes",
            boundaries.len()
        );
        self.boundaries = match boundaries.iter().any(Option::is_some) {
            true => SmallVec::from_slice(boundaries),
            false => SmallVec::new(),
        };
        self
    }

    /// Whether this operand has bounds checking enabled.
    pub fn is_checked(&self) -> bool {
        self.boundaries.iter().any(Option::is_some)
    }

    /// Set the launch's cube size (units per cube).
    pub fn units(mut self, units: usize) -> Self {
        self.units = units;
        self
    }
}
