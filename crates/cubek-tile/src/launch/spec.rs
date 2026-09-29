//! The comptime half of an operand: which axes of the kernel's one [`Space`](crate::Space) its
//! buffer spans and how they address its physical axes.

use cubecl::zspace::SmallVec;

use crate::Delivery;
use crate::{Axis, Boundary, BoundaryPolicy, Field, Packing, Projection, Space, Storage};

/// The comptime half of an operand: how its logical axes address its buffer ([`Projection`], the
/// storage tiling in its own repetition), how its edges are checked, how its values sit in words,
/// what its storage tiles are to the windows it is read through, and who moves it into a stage.
/// [`GlobalOperand::tile`](crate::GlobalOperand::tile) projects the kernel's space onto it, so no operand carries a copy.
///
/// Written by the [`Arg`](crate::Arg) builder's derivation; a kernel that states one by hand
/// ([`new`](Self::new), [`direct`](Self::direct)) fills the fields it needs.
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
    /// of the values in it.
    pub packing: Packing,
    /// What this operand's storage tiles are to the windows it is read through;
    /// [`Strided`](Storage::Strided) for every untiled operand.
    pub storage: Storage,
    /// Who moves this operand into a stage ([`SyncPerUnit`](Delivery::SyncPerUnit) unless stated):
    /// [`delivery`](Self::delivery).
    pub delivery: Delivery,
}

impl TileSpec {
    /// An operand's spec from its mapping, unchecked, plain, untiled, cube size unknown.
    /// [`Projection::validate`] does not run here: its rule on the innermost identity needs the
    /// served width, which only the tile's construction ([`GlobalOperand`](crate::GlobalOperand)) knows.
    pub fn new(projection: Projection) -> Self {
        TileSpec {
            projection,
            boundaries: SmallVec::new(),
            units: 0,
            packing: Packing::Plain,
            storage: Storage::Strided,
            delivery: Delivery::SyncPerUnit,
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

    /// This spec with the boundary mode of every coordinate axis stated; an all-`None` list
    /// collapses to the empty one, so "nothing is checked" has one representation. Shaped over the
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

    /// This spec checked as `policy` says on every coordinate axis. A spec stated by hand has no
    /// partitioning to derive a check from, so [`BoundaryPolicy::Derived`] is refused here.
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

    /// This spec with its binding holding `u32` words packing several values each, one per
    /// `field`-wide slot, innermost axis first; the tile unpacks at the read.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        self.packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// This spec with its stages moved by `delivery`: each unit its own lines, landed when the fill
    /// returns or after it, or one unit the whole stage in bulk. The kernel does not change; the
    /// fill does.
    ///
    /// # Panics
    ///
    /// [`Tma`](Delivery::Tma), which moves a tensor map rather than a tensor, and
    /// [`Procedural`](Delivery::Procedural), which moves nothing.
    pub fn delivery(mut self, delivery: Delivery) -> Self {
        assert!(
            matches!(
                delivery,
                Delivery::SyncPerUnit | Delivery::AsyncPerUnit | Delivery::AsyncBulk
            ),
            "TileSpec::delivery: a tensor is moved per unit or in bulk; {delivery:?} is a tensor \
             map's or a coordinate tile's"
        );
        self.delivery = delivery;
        self
    }

    /// Whether this operand has bounds checking enabled.
    pub fn is_checked(&self) -> bool {
        self.boundaries.iter().any(Option::is_some)
    }
}
