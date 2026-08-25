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

/// How much of the tile geometry space one run sweeps.
///
/// Read from `CUBEK_BENCH_TIER`, alongside the `CUBEK_BENCH_SAMPLES` the harness already takes.
/// The default is [`Light`](BenchTier::Light): a full sweep is 69 geometries over every problem,
/// which is hours, and the geometry that wins is reachable from a fraction of them.
///
/// The tiers nest, so a wider one only ever adds: `Light ⊂ Extensive ⊂ Full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BenchTier {
    /// Enough geometry to pick a config. Measured against the recorded sweeps, it costs 0.6% of
    /// the best CUDA time and 35% of the best CPU time.
    #[default]
    Light,
    /// The tier that finds the best config on every recorded problem. Lossless on the wgpu sweep
    /// and on the CPU sweep, at 28 and 48 geometries respectively.
    Extensive,
    /// Every geometry, both residences. What a new device is characterized with.
    Full,
}

impl BenchTier {
    /// The tier `CUBEK_BENCH_TIER` names. An unset or unrecognized value is [`Light`](Self::Light),
    /// so a harness that knows nothing about tiers runs the cheap sweep rather than the long one.
    pub fn from_env() -> Self {
        match std::env::var("CUBEK_BENCH_TIER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "extensive" => Self::Extensive,
            "full" => Self::Full,
            _ => Self::Light,
        }
    }
}

/// Which device's catalogue to build.
///
/// The two do not want the same sweep. `c > 1` is a lane's column run, which vectorizes a CPU's
/// inner loop and is worth 1.32x there; on a GPU the channel axis already fills the lanes, and
/// dropping every `c > 1` geometry costs 0.6% on CUDA and nothing on wgpu. Deep row runs split the
/// same way. A staged input is refused outright on CPU by
/// [`bench`](super::bench), so a CPU catalogue that offered it would spend half its entries
/// producing the same error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchTarget {
    Gpu,
    Cpu,
}

impl BenchTarget {
    /// The residences worth launching on this device.
    fn residences(self) -> &'static [Residence] {
        match self {
            BenchTarget::Gpu => &[Residence::Smem, Residence::InPlace],
            BenchTarget::Cpu => &[Residence::InPlace],
        }
    }

    /// The `(planes, rows, cols)` geometries this device sweeps at `tier`.
    fn geometries(self, tier: BenchTier) -> &'static [(usize, usize, usize)] {
        match (self, tier) {
            (_, BenchTier::Full) => FULL,
            (BenchTarget::Gpu, BenchTier::Light) => GPU_LIGHT,
            (BenchTarget::Gpu, BenchTier::Extensive) => GPU_EXTENSIVE,
            (BenchTarget::Cpu, BenchTier::Light) => CPU_LIGHT,
            (BenchTarget::Cpu, BenchTier::Extensive) => CPU_EXTENSIVE,
        }
    }
}

/// Planes and rows over the range that ever wins on a GPU, at the one column width that does.
const GPU_LIGHT: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 2, 1),
    (1, 4, 1),
    (1, 8, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 4, 1),
    (4, 1, 1),
    (4, 2, 1),
    (8, 1, 1),
];

/// [`GPU_LIGHT`] completed to the full `p, r in 1..8` core, plus the row-unrolling and
/// plane-count extremes that carry the remaining 13% on the recorded wgpu sweep.
const GPU_EXTENSIVE: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 2, 1),
    (1, 4, 1),
    (1, 8, 1),
    (1, 16, 1),
    (1, 32, 1),
    (1, 64, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 4, 1),
    (2, 8, 1),
    (2, 16, 1),
    (2, 32, 1),
    (4, 1, 1),
    (4, 2, 1),
    (4, 4, 1),
    (4, 8, 1),
    (4, 16, 1),
    (4, 32, 1),
    (8, 1, 1),
    (8, 2, 1),
    (8, 4, 1),
    (8, 8, 1),
    (8, 16, 1),
    (8, 32, 1),
    (16, 1, 1),
    (16, 2, 1),
    (16, 4, 1),
];

/// [`GPU_LIGHT`] plus the deep row runs and the cache-line column widths a CPU needs: the two
/// axes that separate the recorded CPU winners from the rest.
const CPU_LIGHT: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 2, 1),
    (1, 4, 1),
    (1, 4, 2),
    (1, 8, 1),
    (1, 16, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 4, 1),
    (2, 16, 1),
    (4, 1, 1),
    (4, 2, 1),
    (4, 8, 2),
    (4, 16, 1),
    (4, 16, 2),
    (8, 1, 1),
];

/// [`GPU_EXTENSIVE`] with the column sweep a CPU vectorizes over.
const CPU_EXTENSIVE: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 4),
    (1, 1, 8),
    (1, 2, 1),
    (1, 2, 2),
    (1, 2, 4),
    (1, 2, 8),
    (1, 4, 1),
    (1, 4, 2),
    (1, 4, 4),
    (1, 4, 8),
    (1, 8, 1),
    (1, 8, 2),
    (1, 8, 4),
    (1, 8, 8),
    (1, 16, 1),
    (1, 16, 2),
    (1, 16, 4),
    (1, 16, 8),
    (1, 32, 1),
    (1, 64, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 4, 1),
    (2, 8, 1),
    (2, 16, 1),
    (2, 16, 2),
    (2, 32, 1),
    (4, 1, 1),
    (4, 2, 1),
    (4, 4, 1),
    (4, 8, 1),
    (4, 8, 2),
    (4, 16, 1),
    (4, 16, 2),
    (4, 32, 1),
    (8, 1, 1),
    (8, 2, 1),
    (8, 4, 1),
    (8, 8, 1),
    (8, 8, 2),
    (8, 16, 1),
    (8, 16, 4),
    (8, 32, 1),
    (16, 1, 1),
    (16, 2, 1),
    (16, 4, 1),
];

/// Every geometry the catalogue knows, which is what [`BenchTier::Full`] sweeps.
///
/// Instead of a full Cartesian product that explores dead combinations (such as high p × high r ×
/// high c that immediately exceed hardware limits or register files), it covers:
/// - A balanced core sweep across standard dimensions (p, r in 1..8, c=1)
/// - Extreme row-unrolling (high `r`: 16, 32, 64) with low `c` and modest `p`
/// - High cube parallelism (high `p`: 16, 32) with low `r` and `c=1`
/// - Wide column runs (high `c`: 2..16) with small `p` and `r` (primarily for CPU evaluation)
const FULL: &[(usize, usize, usize)] = &[
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
    // Long row runs at a cache-line column width
    (4, 8, 2),
    (8, 8, 2),
    (2, 16, 2),
    (4, 16, 2),
    (2, 32, 2),
    (1, 16, 4),
    (2, 16, 4),
    (4, 16, 4),
    (8, 16, 4),
    (1, 32, 4),
    (4, 8, 4),
    (8, 8, 4),
    (1, 8, 8),
    (2, 8, 8),
    (4, 8, 8),
    (1, 16, 8),
    (2, 16, 8),
];

/// The two baselines the tile path is measured against. Present at every tier, since a tile time
/// with nothing to compare it to says nothing.
fn baselines() -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
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
    ]
}

fn tile_entry(
    residence: Residence,
    (planes, rows, cols): (usize, usize, usize),
) -> CatalogEntry<InterpolateBenchmarkStrategy> {
    let (tag, label) = match residence {
        Residence::Smem => ("smem", "staged"),
        _ => ("in_place", "in-place"),
    };
    CatalogEntry::new(
        format!("tile_{tag}_p{planes}_r{rows}_c{cols}"),
        format!("Tile {label} (p={planes}, r={rows}, c={cols})"),
        InterpolateBenchmarkStrategy::Tile(TileConfig::new(residence, planes, rows, cols)),
    )
}

/// The catalogue at a stated tier and target.
pub fn strategies_at(
    tier: BenchTier,
    target: BenchTarget,
) -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
    let mut entries = baselines();
    for &residence in target.residences() {
        for &geometry in target.geometries(tier) {
            entries.push(tile_entry(residence, geometry));
        }
    }
    entries
}

/// The catalogue a bench run sweeps: the tier named by `CUBEK_BENCH_TIER`.
pub fn strategies(target: BenchTarget) -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
    strategies_at(BenchTier::from_env(), target)
}

/// Every strategy the catalogue can name, whatever tier a run sweeps.
///
/// Lookup by id goes through here, so a correctness test naming a geometry keeps resolving when
/// the tier narrows what a bench run measures.
pub fn every_strategy() -> Vec<CatalogEntry<InterpolateBenchmarkStrategy>> {
    strategies_at(BenchTier::Full, BenchTarget::Gpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const TIERS: [BenchTier; 3] = [BenchTier::Light, BenchTier::Extensive, BenchTier::Full];
    const TARGETS: [BenchTarget; 2] = [BenchTarget::Gpu, BenchTarget::Cpu];

    fn geometries(target: BenchTarget, tier: BenchTier) -> HashSet<(usize, usize, usize)> {
        target.geometries(tier).iter().copied().collect()
    }

    /// A wider tier only ever adds, so a config that wins at one tier is still reachable at the
    /// next. Without it, "extensive" could quietly drop the geometry "light" picked.
    #[test]
    fn the_tiers_nest() {
        for target in TARGETS {
            let light = geometries(target, BenchTier::Light);
            let extensive = geometries(target, BenchTier::Extensive);
            let full = geometries(target, BenchTier::Full);
            assert!(light.is_subset(&extensive), "{target:?}: light ⊄ extensive");
            assert!(extensive.is_subset(&full), "{target:?}: extensive ⊄ full");
        }
    }

    /// `Full` is the whole catalogue on either device, so a characterization run is the same sweep
    /// wherever it happens.
    #[test]
    fn the_full_tier_is_every_geometry() {
        for target in TARGETS {
            assert_eq!(target.geometries(BenchTier::Full).len(), FULL.len());
        }
    }

    /// Ids name a run in the parsed output, so a collision would silently merge two rows.
    #[test]
    fn every_entry_is_named_once() {
        for tier in TIERS {
            for target in TARGETS {
                let entries = strategies_at(tier, target);
                let ids: HashSet<_> = entries.iter().map(|entry| entry.id.clone()).collect();
                assert_eq!(ids.len(), entries.len(), "{tier:?}/{target:?}");
            }
        }
    }

    /// Every tier's entries are nameable, which is what lets a correctness test look one up by id
    /// after a bench run reported it.
    #[test]
    fn every_tier_is_a_subset_of_the_nameable_catalogue() {
        let nameable: HashSet<_> = every_strategy().into_iter().map(|e| e.id).collect();
        for tier in TIERS {
            for target in TARGETS {
                for entry in strategies_at(tier, target) {
                    assert!(nameable.contains(&entry.id), "{} is unnameable", entry.id);
                }
            }
        }
    }

    /// A staged input is refused on CPU, so offering it would spend half the catalogue on one
    /// error message.
    #[test]
    fn the_cpu_catalogue_stages_nothing() {
        for tier in TIERS {
            for entry in strategies_at(tier, BenchTarget::Cpu) {
                if let InterpolateBenchmarkStrategy::Tile(config) = entry.value {
                    assert_ne!(config.input_residence, Residence::Smem, "{}", entry.id);
                }
            }
        }
    }

    /// The baselines are what a tile time is read against, so no tier may drop them.
    #[test]
    fn every_tier_keeps_the_baselines() {
        for tier in TIERS {
            for target in TARGETS {
                let ids: HashSet<_> = strategies_at(tier, target)
                    .into_iter()
                    .map(|e| e.id)
                    .collect();
                assert!(ids.contains("global_memory"), "{tier:?}/{target:?}");
                assert!(ids.contains("shared_memory"), "{tier:?}/{target:?}");
            }
        }
    }
}
