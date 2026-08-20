use cubek_test_utils::CatalogEntry;
use cubek_tile::Residence;

use crate::{
    definition::TileSize,
    launch::{InterpolateStrategy, TileConfig},
    routines::{BlueprintStrategy, GlobalMemoryStrategy, SharedMemoryStrategy},
};

/// The established interpolation implementations and the experimental tile path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolateBenchmarkStrategy {
    Standard(InterpolateStrategy),
    Tile(TileConfig),
}

pub fn strategies() -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
    vec![
        // Standard baseline strategies
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
        // Geometry derives the stable choices: four planes per cube, about 32 columns per cube,
        // and one row per plane when downsampling. Only residence remains worth comparing.
        CatalogEntry::new(
            "tile_auto",
            "Tile gather-reduce (auto residence)",
            InterpolateBenchmarkStrategy::Tile(TileConfig::auto()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_only_compares_residence_for_the_derived_geometry() {
        let ids: Vec<_> = strategies().into_iter().map(|entry| entry.id).collect();
        assert_eq!(
            ids,
            [
                "global_memory",
                "shared_memory",
                "tile_auto",
                "tile_smem",
                "tile_in_place"
            ]
        );
    }
}
