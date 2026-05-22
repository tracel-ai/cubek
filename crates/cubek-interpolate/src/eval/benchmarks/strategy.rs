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
                GlobalMemoryStrategy {},
            )),
        ),
        CatalogEntry::new(
            "shared_memory",
            "Shared Memory",
            InterpolateStrategy::SharedMemoryStrategy(BlueprintStrategy::Inferred(
                SharedMemoryStrategy {
                    shared_memory_height: 1,
                },
            )),
        ),
    ]
}
