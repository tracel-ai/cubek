//! The comptime half of an operand.

use cubecl::zspace::SmallVec;

use crate::{Axis, Boundary, BoundaryPolicy, Field, Packing, Projection, Space, Storage};

/// The comptime half of an operand: its axis mapping, edge checks, packing and storage.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileSpec {
    /// How this operand's logical axes address its buffer's physical ones.
    pub projection: Projection,
    /// Out-of-bounds handling per coordinate axis; `None` or an empty list is unchecked.
    pub boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    /// The launch's units per cube, `0` when unknown.
    pub units: usize,
    /// How this operand's values sit in its binding.
    pub(crate) packing: Packing,
    /// What this operand's storage tiles are to the windows it is read through.
    pub storage: Storage,
}

impl TileSpec {
    /// An operand's spec from its mapping, unchecked, plain, untiled, cube size unknown.
    pub fn new(projection: Projection) -> Self {
        TileSpec {
            projection,
            boundaries: SmallVec::new(),
            units: 0,
            packing: Packing::Plain,
            storage: Storage::Strided,
        }
    }

    /// [`new`](Self::new) over the [`direct`](Projection::direct) mapping.
    pub fn direct(axes: &[Axis]) -> Self {
        Self::new(Projection::direct(axes))
    }

    /// The axes this operand spans, in its logical order.
    pub fn axes(&self) -> &[Axis] {
        self.projection.logical_axes()
    }

    /// This spec with one boundary mode per coordinate axis; panics on any other count.
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

    /// This spec checked as `policy` says; panics on [`BoundaryPolicy::Derived`].
    pub fn boundary(self, policy: BoundaryPolicy) -> Self {
        let coord_rank = self.projection.coordinate_rank();
        match policy {
            BoundaryPolicy::Unchecked => self.boundaries(&vec![None; coord_rank]),
            BoundaryPolicy::Every(boundary) => self.boundaries(&vec![Some(boundary); coord_rank]),
            BoundaryPolicy::Derived => {
                panic!("TileSpec::boundary: a spec stated by hand derives no check; state the mode")
            }
        }
    }

    /// This spec with its binding holding `u32` words of `field`-wide values.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        self.packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// Whether this operand has bounds checking enabled.
    pub fn is_checked(&self) -> bool {
        self.boundaries.iter().any(Option::is_some)
    }
}
