use cubek::convolution::{AcceleratedTileKind, ConvAlgorithm, Strategy};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_SIMPLE_SYNC_CYCLIC_CMMA: &str = "simple_sync_cyclic_cmma";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: STRATEGY_SIMPLE_SYNC_CYCLIC_CMMA.to_string(),
        label: "SimpleSyncCyclic / Cmma (inferred)".to_string(),
    }]
}

pub(crate) fn strategy_for(id: &str) -> Option<Strategy> {
    match id {
        STRATEGY_SIMPLE_SYNC_CYCLIC_CMMA => Some(Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleSyncCyclic,
            tile_kind: AcceleratedTileKind::Cmma,
        }),
        _ => None,
    }
}
