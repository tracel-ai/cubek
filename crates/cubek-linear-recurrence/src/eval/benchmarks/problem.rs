use crate::kernels::walk::RecurrenceWalkStateLayout;
use core::num::NonZeroUsize;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::linear_recurrence::walk::UpdateRule;
use cubek_test_utils::CatalogEntry;

/// One walk's shape: `batch` sequences of `positions` positions over `heads`, keeping the last
/// state, its states owned or in the slots of a store.
#[derive(Debug, Clone, Copy)]
pub struct RecurrenceWalkSpec {
    pub batch: usize,
    pub positions: usize,
    pub heads: RecurrenceHeads,
    pub rule: UpdateRule,
    pub state: RecurrenceWalkStateLayout,
}

/// The slots of the store the in place rows rotate through: 3 MiB each, past the L2 of the cards
/// benched, so a row reads its state cold.
const STORE_SLOTS: usize = 32;

/// A decode step, a short prompt and a long one of a layer with 16 key heads of 3 value heads
/// each, over states of 128 by 128, and a decode step of one row and of eight in the slots of a
/// store; then the additive rule over states of 128 by 64, whose key heads are few and read by
/// many value heads: a decode step, a long prompt and a decode step of eight rows in the slots of
/// a store at one key head of 32 value heads, and a long prompt at 8 key heads of 4.
pub fn problems() -> Vec<CatalogEntry<RecurrenceWalkSpec>> {
    let dense = RecurrenceHeads {
        key_heads: 16,
        heads_per_group: 3,
        key_features: 128,
        value_features: 128,
    };
    let selective = RecurrenceHeads {
        key_heads: 1,
        heads_per_group: 32,
        key_features: 128,
        value_features: 64,
    };
    let grouped = RecurrenceHeads {
        key_heads: 8,
        heads_per_group: 4,
        ..selective
    };
    let owned = |positions, heads, rule| RecurrenceWalkSpec {
        batch: 1,
        positions,
        heads,
        rule,
        state: RecurrenceWalkStateLayout::Owned {
            kept: NonZeroUsize::MIN,
        },
    };
    let in_slots = |batch, heads, rule| RecurrenceWalkSpec {
        batch,
        positions: 1,
        heads,
        rule,
        state: RecurrenceWalkStateLayout::Slots {
            slots: NonZeroUsize::new(STORE_SLOTS).unwrap(),
        },
    };
    let delta = |positions| owned(positions, dense, UpdateRule::Delta);
    let additive = |positions, heads| owned(positions, heads, UpdateRule::Additive);
    vec![
        CatalogEntry::new("t1", "16x3 heads of 128x128, 1 position", delta(1)),
        CatalogEntry::new("t8", "16x3 heads of 128x128, 8 positions", delta(8)),
        CatalogEntry::new("t512", "16x3 heads of 128x128, 512 positions", delta(512)),
        CatalogEntry::new(
            "slots_b1",
            "16x3 heads of 128x128, 1 position, 1 row in 32 slots",
            in_slots(1, dense, UpdateRule::Delta),
        ),
        CatalogEntry::new(
            "slots_b8",
            "16x3 heads of 128x128, 1 position, 8 rows in 32 slots",
            in_slots(8, dense, UpdateRule::Delta),
        ),
        CatalogEntry::new(
            "additive_t1",
            "additive, 1x32 heads of 128x64, 1 position",
            additive(1, selective),
        ),
        CatalogEntry::new(
            "additive_t512",
            "additive, 1x32 heads of 128x64, 512 positions",
            additive(512, selective),
        ),
        CatalogEntry::new(
            "additive_groups8_t512",
            "additive, 8x4 heads of 128x64, 512 positions",
            additive(512, grouped),
        ),
        CatalogEntry::new(
            "additive_slots_b8",
            "additive, 1x32 heads of 128x64, 1 position, 8 rows in 32 slots",
            in_slots(8, selective, UpdateRule::Additive),
        ),
    ]
}
