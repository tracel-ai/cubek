use cubek::convolution::{AcceleratedTileKind, ConvAlgorithm, Strategy};

use crate::registry::CatalogEntry;

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
