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
/// The default is [`Light`](BenchTier::Light): a full sweep is well over a hundred geometries
/// per problem, which is hours, and the geometry that wins is reachable from a fraction of them.
///
/// The tiers nest, so a wider one only ever adds: `Light ⊂ Extensive ⊂ Full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BenchTier {
    /// Enough geometry to pick a config. Measured against the recorded sweeps, it costs 0.6% of
    /// the best CUDA time and 35% of the best CPU time.
    #[default]
    Light,
    /// The tier that finds the best config on every recorded problem: every geometry the recorded
    /// wgpu and CPU sweeps ever won with lies inside this box.
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

    /// The bounds this device sweeps within at `tier`.
    fn sweep(self, tier: BenchTier) -> Sweep {
        match (self, tier) {
            (_, BenchTier::Full) => Sweep::new(32, 64, 16, 512),
            (BenchTarget::Gpu, BenchTier::Light) => Sweep::new(8, 8, 1, 8),
            (BenchTarget::Gpu, BenchTier::Extensive) => Sweep::new(16, 64, 1, 256),
            (BenchTarget::Cpu, BenchTier::Light) => Sweep::new(8, 16, 2, 128),
            (BenchTarget::Cpu, BenchTier::Extensive) => Sweep::new(16, 64, 8, 512),
        }
    }

    /// The `(planes, rows, cols)` geometries this device sweeps at `tier`.
    fn geometries(self, tier: BenchTier) -> Vec<(usize, usize, usize)> {
        self.sweep(tier).geometries()
    }
}

/// The bounds one tier sweeps within, as the corners of a power-of-two box.
///
/// The geometries are the product of that box rather than a list, because the axes are not
/// independent knobs a reader can reason about separately: `planes * rows * cols` is the output a
/// cube holds live, so the three trade against one another and only their product is capped.
/// A tier is therefore four numbers, and widening one is a deliberate edit rather than a
/// hand-extended list whose rule has to be inferred.
///
/// Bounds are inclusive and every extent is a power of two. `units` caps the product, which is
/// what keeps the corner of the box (deep rows at a high plane count) from being swept at all:
/// those geometries exceed the register file and would only ever be refused at launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sweep {
    planes: usize,
    rows: usize,
    cols: usize,
    units: usize,
}

impl Sweep {
    const fn new(planes: usize, rows: usize, cols: usize, units: usize) -> Self {
        Self {
            planes,
            rows,
            cols,
            units,
        }
    }

    /// Every geometry inside the box, in a stable order so catalogue ids do not move between runs.
    fn geometries(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for planes in powers_of_two(self.planes) {
            for rows in powers_of_two(self.rows) {
                for cols in powers_of_two(self.cols) {
                    if planes * rows * cols <= self.units {
                        out.push((planes, rows, cols));
                    }
                }
            }
        }
        out
    }
}

fn powers_of_two(max: usize) -> impl Iterator<Item = usize> {
    core::iter::successors(Some(1usize), move |n| (n * 2 <= max).then_some(n * 2))
}

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
        for geometry in target.geometries(tier) {
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

    /// Whether `outer` contains `inner` whole, which is what makes the tiers nest.
    fn contains(outer: Sweep, inner: Sweep) -> bool {
        inner.planes <= outer.planes
            && inner.rows <= outer.rows
            && inner.cols <= outer.cols
            && inner.units <= outer.units
    }

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

    /// `Full` is the whole box on either device, so a characterization run is the same sweep
    /// wherever it happens.
    #[test]
    fn the_full_tier_is_every_geometry() {
        let gpu = BenchTarget::Gpu.sweep(BenchTier::Full);
        assert_eq!(gpu, BenchTarget::Cpu.sweep(BenchTier::Full));
        assert_eq!(
            geometries(BenchTarget::Gpu, BenchTier::Full),
            geometries(BenchTarget::Cpu, BenchTier::Full)
        );
    }

    /// Nesting is a property of the boxes, not of the geometries they happen to produce: a
    /// widened cap is what the next tier is, so the containment holds before anything is expanded.
    #[test]
    fn the_tier_bounds_nest() {
        for target in TARGETS {
            let light = target.sweep(BenchTier::Light);
            let extensive = target.sweep(BenchTier::Extensive);
            let full = target.sweep(BenchTier::Full);
            assert!(contains(extensive, light), "{target:?}: light ⊄ extensive");
            assert!(contains(full, extensive), "{target:?}: extensive ⊄ full");
        }
    }

    /// Every extent a geometry names is a power of two and inside its tier's box. The catalogue
    /// ids are built from these numbers, so a stray extent would mint an entry nothing can look up.
    #[test]
    fn every_geometry_lies_inside_its_box() {
        for tier in TIERS {
            for target in TARGETS {
                let sweep = target.sweep(tier);
                for (planes, rows, cols) in target.geometries(tier) {
                    for extent in [planes, rows, cols] {
                        assert!(extent.is_power_of_two(), "{extent} is not a power of two");
                    }
                    assert!(planes <= sweep.planes && rows <= sweep.rows && cols <= sweep.cols);
                    assert!(planes * rows * cols <= sweep.units, "{tier:?}/{target:?}");
                }
            }
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
