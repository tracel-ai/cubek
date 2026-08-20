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

/// The tile choices the catalogue sweeps, as the full product of the four.
///
/// Nothing here is filtered on expected merit: a shape that loses everywhere still says where the
/// cliff is, and the derivation these replaced was tuned against a set too small to separate the
/// regimes. What the device cannot serve is refused at launch instead ([`bench`](super::bench)),
/// so an entry that overruns shared memory or a cube's unit budget reports that rather than
/// silently falling back to something else.
const RESIDENCES: [Residence; 2] = [Residence::Smem, Residence::InPlace];
const PLANES_PER_CUBE: [usize; 4] = [1, 2, 4, 8];
const ROWS_PER_PLANE: [usize; 4] = [1, 2, 4, 8];
const COLS_PER_LANE: [usize; 4] = [1, 2, 4, 8];

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
        for planes in PLANES_PER_CUBE {
            for rows in ROWS_PER_PLANE {
                for cols in COLS_PER_LANE {
                    let (tag, label) = match residence {
                        Residence::Smem => ("smem", "staged"),
                        _ => ("in_place", "in-place"),
                    };
                    entries.push(CatalogEntry::new(
                        format!("tile_{tag}_p{planes}_r{rows}_c{cols}"),
                        format!("Tile {label} (p={planes}, r={rows}, c={cols})"),
                        InterpolateBenchmarkStrategy::Tile(TileConfig::new(
                            residence, planes, rows, cols,
                        )),
                    ));
                }
            }
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
        assert_eq!(
            tiles,
            RESIDENCES.len() * PLANES_PER_CUBE.len() * ROWS_PER_PLANE.len() * COLS_PER_LANE.len()
        );
        assert_eq!(entries.len(), tiles + 2, "plus the two baselines");
    }

    /// Ids name a run in the parsed output, so a collision would silently merge two rows.
    #[test]
    fn every_entry_is_named_once() {
        let ids: HashSet<_> = strategies().into_iter().map(|entry| entry.id).collect();
        assert_eq!(ids.len(), strategies().len());
    }
}
