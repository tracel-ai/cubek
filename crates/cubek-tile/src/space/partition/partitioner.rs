//! The [`Partitioner`]: a recursive descent strategy for a [`Space`](crate::Space),
//! one decomposition level plus the partitioner for the subspaces it produces.

use crate::{Axis, ByAxis};

use super::{ComputeScope, Coverage, Distribution, Spatial};

/// A space holds exactly one; [`divide`](crate::Space::divide) consumes the level and
/// hands [`next`](Partitioner::next) down. `Final` carries nothing: what runs on the terminal
/// tile is the kernel's own call.
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
pub(crate) enum LevelScope {
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
pub(crate) enum LevelRole {
    /// Spreads its tiles across hardware instances (`Spatial` on some axis).
    Instance,
    /// Partitions its tiles sequentially across a grid (every axis `Sequential`).
    Partition,
}

/// Several axes' work distributed as one.
///
/// Dealing each axis on its own gives an instance the product of its per-axis runs, which is a
/// box of the grid. These axes are read as a single index instead, so an instance takes a share
/// of the whole rather than a box of it: the shares that no box can describe are exactly the ones
/// that balance a grid its shape cannot divide.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Work {
    axes: Vec<Axis>,
    dist: Spatial,
}

impl Work {
    pub(crate) fn new(axes: Vec<Axis>, dist: Spatial) -> Self {
        Work { axes, dist }
    }

    /// The axes read as one index.
    pub(crate) fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub(crate) fn scope(&self) -> ComputeScope {
        self.dist.scope()
    }

    /// How many instances share the work. Pinned, not derived: the index's length is the whole
    /// level's grid and can be runtime, so nothing here could divide it.
    pub(crate) fn instances(&self) -> usize {
        match self.dist.coverage() {
            Coverage::Instances(n) => n,
            Coverage::TilesEach(_) => panic!(
                "LevelCuts::distribute: state how many instances share the work \
                 (`.instances(n)`); a share of the whole cannot be derived from a grid whose \
                 length is only known at launch"
            ),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    edges: ByAxis<usize>,
    dists: ByAxis<Distribution>,
    scope: LevelScope,
    work: Option<Work>,
    next: Partitioner,
}

impl Level {
    pub(crate) fn scope(&self) -> LevelScope {
        self.scope
    }

    pub(crate) fn role(&self) -> LevelRole {
        self.scope.role()
    }

    pub(crate) fn work(&self) -> Option<&Work> {
        self.work.as_ref()
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

    /// The axes this level distributes as one, if any. Panics on
    /// [`Final`](Partitioner::Final), which carries no level.
    pub fn work(&self) -> Option<&Work> {
        self.level().work.as_ref()
    }

    /// How many levels this chain has left.
    pub fn depth(&self) -> usize {
        match self {
            Partitioner::Final => 0,
            Partitioner::Level(level) => 1 + level.next.depth(),
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
                    work,
                    next,
                } = *level;
                Partitioner::Level(Box::new(Level {
                    edges: sub_tile,
                    dists,
                    scope,
                    work,
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

/// A [`Partitioner`] with its split set, one level deep, closed by
/// [`level`](PartitionerBuilder::level) or [`distributing`](PartitionerBuilder::distributing).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PartitionerBuilder {
    sub_tile: ByAxis<usize>,
    dists: ByAxis<Distribution>,
}

impl Partitioner {
    /// One level cutting each axis to `sub_tile` and dealing it per `dists`; declared axis order,
    /// last axis fastest.
    pub fn over(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>) -> PartitionerBuilder {
        PartitionerBuilder { sub_tile, dists }
    }
}

impl PartitionerBuilder {
    /// Close the level. [`next`](Partitioner::next) is [`Final`](Partitioner::Final) until levels
    /// are stacked with [`with_partitioner`](crate::Space::with_partitioner).
    pub fn level(self) -> Partitioner {
        self.distributing(None)
    }

    /// [`level`](Self::level) for a level that distributes some of its axes' work as one
    /// ([`Work`]).
    pub fn distributing(self, work: Option<Work>) -> Partitioner {
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
            work,
            next: Partitioner::Final,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axis, CubeAxis, Tiling, cubes, lanes, planes};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);

    #[test]
    fn a_level_with_no_spatial_axis_is_sequential() {
        let space = Tiling::over(&[(M, 8), (N, 8)])
            .level(|l| {
                l.walk(&[(M, 4), (N, 4)]);
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Sequential);
        assert_eq!(space.partitioner().role(), LevelRole::Partition);
    }

    #[test]
    fn cube_axes_alone_separate_what_the_launch_grid_does() {
        let space = Tiling::over(&[(M, 8), (N, 8)])
            .level(|l| {
                l.distribute(cubes(CubeAxis::Y), &[(M, 4)])
                    .distribute(cubes(CubeAxis::X), &[(N, 4)]);
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Cubes);
        assert_eq!(space.partitioner().role(), LevelRole::Instance);
    }

    /// The case the fold exists for: a cube axis beside a plane axis reaches inside a cube, so the
    /// level reads as `Planes`. Reading only the first axis, or the coarsest, would say `Cubes`.
    #[test]
    fn a_plane_axis_beside_a_cube_axis_reaches_inside_the_cube() {
        let space = Tiling::over(&[(M, 8), (N, 8)])
            .level(|l| {
                l.distribute(cubes(CubeAxis::Y), &[(M, 4)])
                    .distribute(planes(), &[(N, 4)]);
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Planes);
    }

    #[test]
    fn a_unit_axis_is_the_finest_scope() {
        let space = Tiling::over(&[(M, 8), (N, 8)])
            .level(|l| {
                l.distribute(planes(), &[(M, 4)])
                    .distribute(lanes(4), &[(N, 4)]);
            })
            .build();
        assert_eq!(space.partitioner().scope(), LevelScope::Lanes);
    }
}
