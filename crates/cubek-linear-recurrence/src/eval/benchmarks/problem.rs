use crate::kernels::walk::RecurrenceWalkStateLayout;
use core::num::NonZeroUsize;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_test_utils::CatalogEntry;

/// One walk's shape: `batch` sequences of `positions` positions over `heads`, keeping the last
/// state, its states owned or in the slots of a store.
#[derive(Debug, Clone, Copy)]
pub struct RecurrenceWalkSpec {
    pub batch: usize,
    pub positions: usize,
    pub heads: RecurrenceHeads,
    pub state: RecurrenceWalkStateLayout,
}

/// The slots of the store the in place rows rotate through: 3 MiB each, past the L2 of the cards
/// benched, so a row reads its state cold.
const STORE_SLOTS: usize = 32;

/// A decode step, a short prompt and a long one of a layer with 16 key heads of 3 value heads
/// each, over states of 128 by 128, and a decode step of one row and of eight in the slots of a
/// store.
pub fn problems() -> Vec<CatalogEntry<RecurrenceWalkSpec>> {
    let heads = RecurrenceHeads {
        key_heads: 16,
        heads_per_group: 3,
        key_features: 128,
        value_features: 128,
    };
    let owned = |positions| RecurrenceWalkSpec {
        batch: 1,
        positions,
        heads,
        state: RecurrenceWalkStateLayout::Owned {
            kept: NonZeroUsize::MIN,
        },
    };
    let in_slots = |batch| RecurrenceWalkSpec {
        batch,
        positions: 1,
        heads,
        state: RecurrenceWalkStateLayout::Slots {
            slots: NonZeroUsize::new(STORE_SLOTS).unwrap(),
        },
    };
    vec![
        CatalogEntry::new("t1", "16x3 heads of 128x128, 1 position", owned(1)),
        CatalogEntry::new("t8", "16x3 heads of 128x128, 8 positions", owned(8)),
        CatalogEntry::new("t512", "16x3 heads of 128x128, 512 positions", owned(512)),
        CatalogEntry::new(
            "slots_b1",
            "16x3 heads of 128x128, 1 position, 1 row in 32 slots",
            in_slots(1),
        ),
        CatalogEntry::new(
            "slots_b8",
            "16x3 heads of 128x128, 1 position, 8 rows in 32 slots",
            in_slots(8),
        ),
    ]
}
