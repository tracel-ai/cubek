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

/// The tile geometries the catalogue evaluates.
///
/// Instead of a full Cartesian product that explores dead combinations (such as high p × high r × high c
/// that immediately exceed hardware limits or register files), the catalogue evaluates:
/// - A balanced core sweep across standard dimensions (p, r in 1..8, c=1)
/// - Extreme row-unrolling (high `r`: 16, 32, 64) with low `c` and modest `p`
/// - High cube parallelism (high `p`: 16, 32) with low `r` and `c=1`
/// - Wide column runs (high `c`: 2..16) with small `p` and `r` (primarily for CPU evaluation)
const RESIDENCES: [Residence; 2] = [Residence::Smem, Residence::InPlace];
const TILE_GEOMETRIES: &[(usize, usize, usize)] = &[
    // Standard baseline sweep (p, r in 1..8, c=1)
    (1, 1, 1),
    (1, 2, 1),
    (1, 4, 1),
    (1, 8, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 4, 1),
    (2, 8, 1),
    (4, 1, 1),
    (4, 2, 1),
    (4, 4, 1),
    (4, 8, 1),
    (8, 1, 1),
    (8, 2, 1),
    (8, 4, 1),
    (8, 8, 1),
    // Extreme row-unrolling (high r: 16, 32, 64) with low c and modest p
    (1, 16, 1),
    (1, 32, 1),
    (1, 64, 1),
    (2, 16, 1),
    (2, 32, 1),
    (2, 64, 1),
    (4, 16, 1),
    (4, 32, 1),
    (8, 16, 1),
    (8, 32, 1),
    // High cube parallelism / plane count (high p: 16, 32) with low r and c=1
    (16, 1, 1),
    (16, 2, 1),
    (16, 4, 1),
    (32, 1, 1),
    (32, 2, 1),
    // Column unrolling / multi-column sweeps (c in 2..16, especially for CPU / vectorization)
    (1, 1, 2),
    (1, 2, 2),
    (1, 4, 2),
    (1, 8, 2),
    (1, 16, 2),
    (2, 2, 2),
    (4, 2, 2),
    (1, 1, 4),
    (1, 2, 4),
    (1, 4, 4),
    (1, 8, 4),
    (2, 2, 4),
    (4, 2, 4),
    (1, 1, 8),
    (1, 2, 8),
    (1, 4, 8),
    (2, 2, 8),
    (1, 1, 16),
    (1, 2, 16),
    (1, 4, 16),
    (2, 1, 16),
];

pub fn strategies() -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
    let mut entries = vec![
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
    ];

    for residence in RESIDENCES {
        for &(planes, rows, cols) in TILE_GEOMETRIES {
            let (tag, label) = match residence {
                Residence::Smem => ("smem", "staged"),
                _ => ("in_place", "in-place"),
            };
            entries.push(CatalogEntry::new(
                format!("tile_{tag}_p{planes}_r{rows}_c{cols}"),
                format!("Tile {label} (p={planes}, r={rows}, c={cols})"),
                InterpolateBenchmarkStrategy::Tile(TileConfig::new(residence, planes, rows, cols)),
            ));
        }
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn the_catalogue_sweeps_every_tile_choice() {
        let entries = strategies();
        let tiles = entries
            .iter()
            .filter(|entry| matches!(entry.value, InterpolateBenchmarkStrategy::Tile(_)))
            .count();
        assert_eq!(tiles, RESIDENCES.len() * TILE_GEOMETRIES.len());
        assert_eq!(entries.len(), tiles + 2, "plus the two baselines");
    }

    /// Ids name a run in the parsed output, so a collision would silently merge two rows.
    #[test]
    fn every_entry_is_named_once() {
        let ids: HashSet<_> = strategies().into_iter().map(|entry| entry.id).collect();
        assert_eq!(ids.len(), strategies().len());
    }
}
