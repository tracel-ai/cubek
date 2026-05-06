use cubek::attention::{
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

use crate::registry::CatalogEntry;

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "unit_inferred",
            "Unit (inferred)",
            Strategy::Unit(BlueprintStrategy::Inferred(())),
        ),
        CatalogEntry::new(
            "blackbox_accelerated_inferred",
            "Blackbox accelerated (inferred, np=1 sq=1 skv=1)",
            Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
                BlackboxAcceleratedStrategy {
                    num_planes: 1,
                    seq_q: 1,
                    seq_kv: 1,
                },
            )),
        ),
    ]
}
