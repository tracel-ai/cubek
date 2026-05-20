use cubek_test_utils::CatalogEntry;

use crate::definition::{InterpolateStrategy, MemoryStrategy};

pub fn strategies() -> Vec<CatalogEntry<InterpolateStrategy>> {
    vec![
        CatalogEntry::new(
            "global_memory",
            "Global Memory",
            InterpolateStrategy::new(MemoryStrategy::Global),
        ),
        CatalogEntry::new(
            "shared_memory",
            "Shared Memory",
            InterpolateStrategy::new(MemoryStrategy::Shared),
        ),
    ]
}
