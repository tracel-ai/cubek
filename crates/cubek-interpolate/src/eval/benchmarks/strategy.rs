use cubek_test_utils::CatalogEntry;
use cubek_tile::Residence;

use crate::{
    definition::TileSize,
    launch::{InterpolateStrategy, TileConfig},
    routines::{BlueprintStrategy, GlobalMemoryStrategy, SharedMemoryStrategy},
};

/// The established interpolation implementations and the experimental tile path.
///
/// The tile path is entered twice, once per input residence, because those are its only two
/// distinct kernels: the ring is single-slot everywhere ([`interpolate_space`]), so a deeper
/// buffering would compile the same shader and time the same noise. `TileConfig::auto` is not
/// listed for the same reason, being whichever of the two the tap window selects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolateBenchmarkStrategy {
    Standard(InterpolateStrategy),
    Tile(TileConfig),
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
            "tile_smem",
            "Tile gather-reduce (staged input)",
            InterpolateBenchmarkStrategy::Tile(TileConfig::forced(Residence::Smem)),
        ),
        CatalogEntry::new(
            "tile_in_place",
            "Tile gather-reduce (in-place input)",
            InterpolateBenchmarkStrategy::Tile(TileConfig::forced(Residence::InPlace)),
        ),
    ]
}
