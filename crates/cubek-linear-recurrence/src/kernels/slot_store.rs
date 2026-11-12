//! A store of slots and the tables that place each row of a launch in it.

use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::tile::{Axis, Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::{BATCH, STORE_SLOT};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

use crate::kernels::bound::bound;

/// A store whose leading axis is its slots, and for each row of a launch the slot it lies in
/// (`slots`, `[rows]`, `u32`) and the positions it has advanced already (`lengths`, `[rows]`,
/// `u32`). Two rows may share a slot only if it is never read, and a slot past the store's last
/// reads and writes the last.
#[derive(Debug, Clone)]
pub struct SlotStore {
    store: PlainTensor,
    slots: PlainTensor,
    lengths: PlainTensor,
}

/// A [`SlotStore`] bound under one launcher.
pub struct BoundSlots {
    /// The store, each row placed at the slot its table names.
    pub store: TileTensorLaunch,
    /// The positions each row has advanced.
    pub lengths: TileTensorLaunch,
}

impl SlotStore {
    /// # Panics
    ///
    /// On tables that are not `u32`, not `[rows]` of one `rows`, and on a store of no slot.
    pub fn new(store: PlainTensor, slots: PlainTensor, lengths: PlainTensor) -> Self {
        let rows = slots.shape().dims::<1>()[0];
        for table in [&slots, &lengths] {
            assert_eq!(table.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
            assert_eq!(table.shape().dims::<1>(), [rows], "a table is [rows]");
        }
        assert!(store.shape().to_vec()[0] > 0, "a store holds a slot");
        Self {
            store,
            slots,
            lengths,
        }
    }

    /// The store.
    pub fn store(&self) -> &PlainTensor {
        &self.store
    }

    /// The positions each row has advanced.
    pub fn lengths(&self) -> &PlainTensor {
        &self.lengths
    }

    /// The rows the tables place.
    pub fn rows(&self) -> usize {
        self.slots.shape().dims::<1>()[0]
    }

    /// The slots the store holds.
    pub fn slot_count(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.store.shape().to_vec()[0]).expect("a store holds a slot")
    }

    /// The same store read as `meta` states.
    pub fn viewed(self, meta: Metadata) -> Self {
        Self {
            store: self.store.viewed(meta),
            ..self
        }
    }

    /// The same store and lengths with `slots` placing the rows.
    pub fn placed(self, slots: PlainTensor) -> Self {
        Self::new(self.store, slots, self.lengths)
    }

    /// The store bound under `launcher` along `store_axes`, `width` values a line, and the
    /// lengths bound along the rows. The only place the slot table indexes the store.
    pub fn bind(&self, launcher: &Launcher, store_axes: &[Axis], width: usize) -> BoundSlots {
        let entries = TileTensorLaunch::new(
            self.slots.binding.clone().into_tensor_arg(),
            self.slots.elem,
            1,
            TileSpec::direct(&[BATCH]),
        );
        BoundSlots {
            store: bound(launcher, &self.store, store_axes, width)
                .indexed(entries, Indexing::Coordinate(STORE_SLOT))
                .expect(ADMITTED_PLAN_RUNS),
            lengths: bound(launcher, &self.lengths, &[BATCH], 1),
        }
    }
}
