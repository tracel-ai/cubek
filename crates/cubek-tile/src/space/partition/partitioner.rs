//! The [`Partitioner`]: the levels a [`Space`](crate::Space) still carries below itself, one
//! [`Level`] plus the partitioner for the subspaces it produces.

use crate::{Axis, ByAxis};

use super::{Distribution, Level, LevelRole, LevelScope, Work};

/// A space holds exactly one; [`divide`](crate::Space::divide) consumes the level and
/// hands [`next`](Partitioner::next) down. `Final` carries nothing: what runs on the terminal
/// tile is the kernel's own call.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Partitioner {
    Final,
    Level(Box<Chain>),
}

/// One [`Level`] and the partitioner below it.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Chain {
    level: Level,
    next: Partitioner,
}

impl Chain {
    pub(crate) fn level(&self) -> &Level {
        &self.level
    }

    pub(crate) fn role(&self) -> LevelRole {
        self.level.role()
    }

    pub(crate) fn scope(&self) -> LevelScope {
        self.level.scope()
    }
}

impl Partitioner {
    /// One level with nothing below it, until levels are stacked with
    /// [`with_partitioner`](crate::Space::with_partitioner).
    pub fn single(level: Level) -> Partitioner {
        Partitioner::Level(Box::new(Chain {
            level,
            next: Partitioner::Final,
        }))
    }

    pub fn is_final(&self) -> bool {
        matches!(self, Partitioner::Final)
    }

    pub fn next(&self) -> &Partitioner {
        &self.chain().next
    }

    /// This partitioner's own level. Panics on [`Final`](Partitioner::Final), which carries
    /// none.
    pub fn level(&self) -> &Level {
        &self.chain().level
    }

    pub fn edge(&self, axis: Axis) -> usize {
        self.level().edge(axis)
    }

    pub fn distribution(&self, axis: Axis) -> Distribution {
        self.level().distribution(axis)
    }

    /// This level's [`LevelScope`]. Panics on [`Final`](Partitioner::Final), which carries no
    /// level.
    pub(crate) fn scope(&self) -> LevelScope {
        self.level().scope()
    }

    /// This level's [`LevelRole`]. Panics on [`Final`](Partitioner::Final), which carries no level.
    pub(crate) fn role(&self) -> LevelRole {
        self.level().role()
    }

    /// The axes this level distributes, which outlive the space they came from: a level keeps
    /// every axis of the operation, so an output space (`{M, N}`) still names its contraction.
    /// Panics on [`Final`](Partitioner::Final), which carries no level.
    pub(crate) fn axes(&self) -> Vec<Axis> {
        self.level().axes()
    }

    /// The axes this level distributes as one, if any. Panics on
    /// [`Final`](Partitioner::Final), which carries no level.
    pub fn work(&self) -> Option<&Work> {
        self.level().work()
    }

    /// How many levels this chain has left.
    pub fn depth(&self) -> usize {
        match self {
            Partitioner::Final => 0,
            Partitioner::Level(chain) => 1 + chain.next.depth(),
        }
    }

    pub(crate) fn append(self, tail: Partitioner) -> Partitioner {
        match self {
            Partitioner::Final => tail,
            Partitioner::Level(chain) => {
                let Chain { level, next } = *chain;
                Partitioner::Level(Box::new(Chain {
                    level,
                    next: next.append(tail),
                }))
            }
        }
    }

    /// Panics on [`Final`](Partitioner::Final), which carries no level.
    fn chain(&self) -> &Chain {
        match self {
            Partitioner::Level(chain) => chain,
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
        Partitioner::single(Level::from_parts(self.sub_tile, self.dists, work))
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
