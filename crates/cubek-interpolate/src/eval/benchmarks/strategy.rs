use cubek_test_utils::CatalogEntry;

use crate::{
    launch::InterpolateStrategy,
    routines::{BlueprintStrategy, GlobalMemoryStrategy, SharedMemoryStrategy},
};

pub fn strategies() -> Vec<CatalogEntry<InterpolateStrategy>> {
    vec![
        CatalogEntry::new(
            "global_memory",
            "Global Memory",
            InterpolateStrategy::GlobalMemoryStrategy(BlueprintStrategy::Inferred(
                GlobalMemoryStrategy {
                    tile_target_aspect_ratio: 1.0,
                },
            )),
        ),
        CatalogEntry::new(
            "shared_memory",
            "Shared Memory",
            InterpolateStrategy::SharedMemoryStrategy(BlueprintStrategy::Inferred(
                SharedMemoryStrategy {
                    tile_target_aspect_ratio: 1.0,
                },
            )),
        ),
    ]
}
