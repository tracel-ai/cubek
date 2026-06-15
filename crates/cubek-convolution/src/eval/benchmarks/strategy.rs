use cubek_test_utils::CatalogEntry;

use crate::{AcceleratedTileKind, ConvAlgorithm, Strategy};

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![CatalogEntry::new(
        "simple_sync_cyclic_cmma",
        "SimpleSyncCyclic / Cmma (inferred)",
        Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleSyncCyclic,
            tile_kind: AcceleratedTileKind::Cmma,
        },
    )]
}
