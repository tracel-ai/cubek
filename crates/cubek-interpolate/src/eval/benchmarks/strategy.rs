use cubek_test_utils::CatalogEntry;

use crate::{
    definition::TileSize,
    launch::InterpolateStrategy,
    routines::{BlueprintStrategy, GlobalMemoryStrategy, SharedMemoryStrategy},
};

/// The established interpolation implementations and the experimental tile path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolateBenchmarkStrategy {
    Standard(InterpolateStrategy),
    Tile,
}

pub fn strategies() -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
    vec![
        CatalogEntry::new(
            "global_memory",
            "Global Memory",
            InterpolateBenchmarkStrategy::Standard(InterpolateStrategy::GlobalMemoryStrategy(
                BlueprintStrategy::Inferred(GlobalMemoryStrategy {
                    tile_size: TileSize::new(16, 16),
                }),
            )),
        ),
        CatalogEntry::new(
            "shared_memory",
            "Shared Memory",
            InterpolateBenchmarkStrategy::Standard(InterpolateStrategy::SharedMemoryStrategy(
                BlueprintStrategy::Inferred(SharedMemoryStrategy {
                    tile_size: TileSize::new(16, 16),
                }),
            )),
        ),
        CatalogEntry::new(
            "tile",
            "Tile gather-reduce",
            InterpolateBenchmarkStrategy::Tile,
        ),
    ]
}
