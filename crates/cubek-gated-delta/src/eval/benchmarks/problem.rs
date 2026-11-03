use cubek_microkernel::gated_delta::GatedDeltaHeads;
use cubek_test_utils::CatalogEntry;

/// One walk's shape: `batch` sequences of `positions` positions over `heads`, keeping the last
/// state.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaWalkSpec {
    pub batch: usize,
    pub positions: usize,
    pub heads: GatedDeltaHeads,
}

/// A decode step, a short prompt and a long one of a layer with 16 key heads of 3 value heads
/// each, over states of 128 by 128.
pub fn problems() -> Vec<CatalogEntry<GatedDeltaWalkSpec>> {
    let heads = GatedDeltaHeads {
        key_heads: 16,
        group: 3,
        key_features: 128,
        value_features: 128,
    };
    let spec = |positions| GatedDeltaWalkSpec {
        batch: 1,
        positions,
        heads,
    };
    vec![
        CatalogEntry::new("t1", "16x3 heads of 128x128, 1 position", spec(1)),
        CatalogEntry::new("t8", "16x3 heads of 128x128, 8 positions", spec(8)),
        CatalogEntry::new("t512", "16x3 heads of 128x128, 512 positions", spec(512)),
    ]
}
