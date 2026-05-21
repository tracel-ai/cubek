use cubek_test_utils::CatalogEntry;

use crate::{
    launch::{InterpolateStrategy, RoutineStrategy},
    routines::{BlueprintStrategy, GlobalMemoryStrategy, SharedMemoryStrategy},
};

pub fn strategies() -> Vec<CatalogEntry<InterpolateStrategy>> {
    vec![
        CatalogEntry::new(
            "global_memory",
            "Global Memory",
            InterpolateStrategy {
                routine: RoutineStrategy::GlobalMemoryStrategy(BlueprintStrategy::Inferred(
                    GlobalMemoryStrategy {},
                )),
            },
        ),
        CatalogEntry::new(
            "shared_memory",
            "Shared Memory",
            InterpolateStrategy {
                routine: RoutineStrategy::SharedMemoryStrategy(BlueprintStrategy::Inferred(
                    SharedMemoryStrategy {
                        shared_memory_height: 1,
                    },
                )),
            },
        ),
    ]
}
