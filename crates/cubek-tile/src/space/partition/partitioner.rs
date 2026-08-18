//! The [`Partitioner`]: a recursive descent strategy for a [`Space`](crate::Space),
//! one decomposition level plus the partitioner for the subspaces it produces.

use crate::{Axis, ByAxis, MmaIOConfig};

use super::{ComputeScope, Distribution, WalkOrder};

/// How deeply a level's walk buffers its regions, and nothing else: whether an operand is
/// materialized at all, and into what, is the operand's own
/// [`Residence`](crate::Residence), stated per level. The two are independent: a level every
/// operand rides [`InPlace`](crate::Residence::InPlace) still runs the depth stated here, over
/// slots that allocate nothing.
///
/// A depth, not a menu: the walk is one circular software pipeline of `depth` slots
/// ([`Ring`](crate::Ring)), so single and double buffering are the `1` and `2` of the same
/// schedule rather than two hand-written ones.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Buffering(usize);

impl Buffering {
    /// One slot: fill a region, consume it, repeat.
    pub const SINGLE: Buffering = Buffering(1);
    /// Two slots alternating, so one region's fill overlaps the other's compute.
    pub const DOUBLE: Buffering = Buffering(2);
    /// Three slots circular, so two fills are in flight over one compute.
    pub const TRIPLE: Buffering = Buffering(3);

    /// A pipeline `depth` slots deep. Panics on `0`, which buffers nothing and computes nothing.
    pub const fn new(depth: usize) -> Self {
        assert!(depth > 0, "Buffering: a pipeline needs at least one slot");
        Buffering(depth)
    }

    /// How many slots the walk drives.
    pub const fn depth(self) -> usize {
        self.0
    }
}

impl Default for Buffering {
    fn default() -> Self {
        Self::SINGLE
    }
}

/// What an operand *is* at the instruction: a memory window, or a plane fragment in one of the two
/// encodings. Pure format, no shape — `m`/`n`/`k` belong to the contraction, not to any one operand,
/// so every allocation site is handed them by whoever holds enough spaces to know them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Leaf {
    /// A memory window contracted by the software microkernel.
    #[default]
    Memory,
    Cmma,
    /// The manual/raw-mma rung: `MmaDefinition::execute` over register fragments. `io` rides the
    /// leaf because it comes from a device query, which cannot run in-kernel.
    Mma {
        io: MmaIOConfig,
    },
}

/// A space holds exactly one; [`divide`](crate::Space::divide) consumes the level and
/// hands [`next`](Partitioner::next) down. A `Level` carries how deeply its walk buffers
/// ([`Buffering`]); `Final` carries how to contract the terminal tile ([`Leaf`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Partitioner {
    Final,
    Level(Box<Level>),
}

/// How finely a level separates its tiles: the smallest hardware scope any of its axes rides.
/// Decided once, when the level is built, so no consumer re-folds the per-axis distributions.
///
/// The finest scope wins. A level with an axis on a cube dim and another on planes reaches
/// inside a cube, and what a level separates inside a cube the cube's own cooperative
/// transports already spread.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum LevelScope {
    /// Every axis `Sequential`: one instance walks the whole grid.
    Sequential,
    /// Some axis rides a cube dim and none reaches inside a cube, so the level separates
    /// exactly what the launch grid does.
    Cubes,
    /// Some axis rides the cube's planes.
    Planes,
    /// Some axis rides a plane's lanes.
    Lanes,
}

impl LevelScope {
    /// The scope one axis's distribution puts a level in.
    fn of(dist: Distribution) -> Self {
        match dist.scope() {
            None => LevelScope::Sequential,
            Some(ComputeScope::Cube(_)) => LevelScope::Cubes,
            Some(ComputeScope::Plane) => LevelScope::Planes,
            Some(ComputeScope::Unit) => LevelScope::Lanes,
        }
    }

    /// The coarse reading, for consumers that only ask whether the level spreads at all.
    pub(crate) fn role(self) -> LevelRole {
        match self {
            LevelScope::Sequential => LevelRole::Partition,
            LevelScope::Cubes | LevelScope::Planes | LevelScope::Lanes => LevelRole::Instance,
        }
    }
}

/// Whether a level spreads its tiles across hardware at all, which is all most consumers ask.
/// A view over [`LevelScope`], never stored: the scope is the level's own state.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LevelRole {
    /// Spreads its tiles across hardware instances (`Spatial` on some axis).
    Instance,
    /// Partitions its tiles sequentially across a grid (every axis `Sequential`).
    Partition,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    edges: ByAxis<usize>,
    dists: ByAxis<Distribution>,
    scope: LevelScope,
    order: WalkOrder,
    buffering: Buffering,
    next: Partitioner,
}

impl Level {
    pub fn buffering(&self) -> Buffering {
        self.buffering
    }

    pub(crate) fn scope(&self) -> LevelScope {
        self.scope
    }

    pub(crate) fn role(&self) -> LevelRole {
        self.scope.role()
    }
}

impl Partitioner {
    pub fn is_final(&self) -> bool {
        matches!(self, Partitioner::Final)
    }

    pub fn next(&self) -> &Partitioner {
        &self.level().next
    }

    pub fn edge(&self, axis: Axis) -> usize {
        self.level().edges.get(axis)
    }

    pub fn distribution(&self, axis: Axis) -> Distribution {
        self.level().dists.get(axis)
    }

    /// This level's [`LevelScope`]. Panics on [`Final`](Partitioner::Final), which carries no
    /// level.
    pub(crate) fn scope(&self) -> LevelScope {
        self.level().scope
    }

    /// This level's [`LevelRole`]. Panics on [`Final`](Partitioner::Final), which carries no level.
    pub(crate) fn role(&self) -> LevelRole {
        self.level().scope.role()
    }

    /// The axes this level distributes, which outlive the space they came from: a level keeps
    /// every axis of the operation, so an output space (`{M, N}`) still names its contraction.
    /// Panics on [`Final`](Partitioner::Final), which carries no level.
    pub(crate) fn axes(&self) -> Vec<Axis> {
        let dists = &self.level().dists;
        (0..dists.len()).map(|i| dists.axis_at(i)).collect()
    }

    pub fn order(&self) -> WalkOrder {
        self.level().order
    }

    /// How many levels this chain has left, i.e. how many residences an operand descending it
    /// states ([`StagePlan`](crate::StagePlan)).
    pub fn depth(&self) -> usize {
        match self {
            Partitioner::Final => 0,
            Partitioner::Level(level) => 1 + level.next.depth(),
        }
    }

    /// Resolve every level's deferred [`PlaneLanes`](super::Coverage::PlaneLanes) count to
    /// `Instances(plane_size)`. The launch's single stamping pass, so geometry and the walk
    /// only ever see concrete instance counts.
    pub(crate) fn resolve_lanes(self, plane_size: usize) -> Partitioner {
        match self {
            Partitioner::Final => Partitioner::Final,
            Partitioner::Level(level) => {
                let Level {
                    edges,
                    dists,
                    scope,
                    order,
                    buffering,
                    next,
                } = *level;
                // Resolving lane counts keeps every axis `Spatial`, so the scope is unchanged.
                Partitioner::Level(Box::new(Level {
                    edges,
                    dists: dists.map(|_, d| d.resolve_lanes(plane_size)),
                    scope,
                    order,
                    buffering,
                    next: next.resolve_lanes(plane_size),
                }))
            }
        }
    }

    pub(crate) fn append(self, tail: Partitioner) -> Partitioner {
        match self {
            Partitioner::Final => tail,
            Partitioner::Level(level) => {
                let Level {
                    edges: sub_tile,
                    dists,
                    scope,
                    order,
                    buffering,
                    next,
                } = *level;
                Partitioner::Level(Box::new(Level {
                    edges: sub_tile,
                    dists,
                    scope,
                    order,
                    buffering,
                    next: next.append(tail),
                }))
            }
        }
    }

    /// Panics on [`Final`](Partitioner::Final), which carries no level.
    fn level(&self) -> &Level {
        match self {
            Partitioner::Level(level) => level,
            Partitioner::Final => {
                panic!(
                    "Partitioner: the final partitioner carries no level (check `is_final` first)"
                )
            }
        }
    }
}

/// A [`Partitioner`] with its split and walk order set but no [`Buffering`] yet.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PartitionerBuilder {
    sub_tile: ByAxis<usize>,
    dists: ByAxis<Distribution>,
    order: WalkOrder,
}

impl PartitionerBuilder {
    pub(super) fn new(
        sub_tile: ByAxis<usize>,
        dists: ByAxis<Distribution>,
        order: WalkOrder,
    ) -> Self {
        PartitionerBuilder {
            sub_tile,
            dists,
            order,
        }
    }

    /// [`next`](Partitioner::next) is [`Final`](Partitioner::Final) until levels are
    /// stacked with [`with_partitioner`](crate::Space::with_partitioner).
    fn finish(self, buffering: Buffering) -> Partitioner {
        // The finest scope any axis rides; `Sequential` when none spreads at all.
        let scope = self
            .dists
            .values()
            .fold(LevelScope::Sequential, |scope, dist| {
                scope.max(LevelScope::of(dist))
            });
        Partitioner::Level(Box::new(Level {
            edges: self.sub_tile,
            dists: self.dists,
            scope,
            order: self.order,
            buffering,
            next: Partitioner::Final,
        }))
    }

    /// Close the level with the depth its walk buffers to. The one way to state it: a depth is a
    /// number, so naming a few of them would only hide that.
    pub fn buffered(self, buffering: Buffering) -> Partitioner {
        self.finish(buffering)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axis, CubeAxis, Cut, Tiling};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);

    #[test]
    fn a_level_with_no_spatial_axis_is_sequential() {
        let space = Tiling::new()
            .extents(&[(M, 8), (N, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, Cut::sequential(4)).axis(N, Cut::sequential(4))
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Sequential);
        assert_eq!(space.partitioner().role(), LevelRole::Partition);
    }

    #[test]
    fn cube_axes_alone_separate_what_the_launch_grid_does() {
        let space = Tiling::new()
            .extents(&[(M, 8), (N, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, Cut::cube(CubeAxis::Y, 4))
                    .axis(N, Cut::cube(CubeAxis::X, 4))
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Cubes);
        assert_eq!(space.partitioner().role(), LevelRole::Instance);
    }

    /// The case the fold exists for: a cube axis beside a plane axis reaches inside a cube, so the
    /// level reads as `Planes`. Reading only the first axis, or the coarsest, would say `Cubes`.
    #[test]
    fn a_plane_axis_beside_a_cube_axis_reaches_inside_the_cube() {
        let space = Tiling::new()
            .extents(&[(M, 8), (N, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, Cut::cube(CubeAxis::Y, 4)).axis(N, Cut::plane(4))
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Planes);
    }

    #[test]
    fn a_unit_axis_is_the_finest_scope() {
        let space = Tiling::new()
            .extents(&[(M, 8), (N, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, Cut::plane(4)).axis(N, Cut::unit(4))
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Lanes);
    }
}
