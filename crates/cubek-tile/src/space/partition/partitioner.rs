//! The [`Partitioner`]: a recursive descent strategy for a [`Space`](crate::Space),
//! one decomposition level plus the partitioner for the subspaces it produces.

use crate::{Axis, ByAxis, MmaIOConfig};

use super::{Distribution, WalkOrder};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Schedule {
    Direct,
    Staged,
    /// Staged with two buffers, prefetching the next sub-tile while computing.
    DoubleBuffered,
}

/// The instruction that contracts a final tile. Declared in the plan because pre-leaf
/// code (residency, staging-store deduction, cmma smem tiling) reads it before the leaf
/// runs. `Cmma` carries the contraction depth `k`, which an accumulator's axes never give.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Leaf {
    #[default]
    Register,
    Cmma {
        k: usize,
    },
    /// The manual/raw-mma rung: `MmaDefinition::execute` over register fragments. `io` rides the
    /// leaf because it comes from a device query, which cannot run in-kernel.
    Mma {
        k: usize,
        io: MmaIOConfig,
    },
}

impl Leaf {
    pub fn is_cmma(&self) -> bool {
        matches!(self, Leaf::Cmma { .. })
    }

    /// Whether the leaf contracts a plane-level tile (either encoding), so operands and the
    /// accumulator are plane-resident rather than memory.
    pub fn is_plane(&self) -> bool {
        matches!(self, Leaf::Cmma { .. } | Leaf::Mma { .. })
    }
}

/// A space holds exactly one; [`divide`](crate::Space::divide) consumes the level and
/// hands [`next`](Partitioner::next) down. A `Level` carries how to walk its regions
/// ([`Schedule`]); `Final` carries how to contract the terminal tile ([`Leaf`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Partitioner {
    Final(Leaf),
    Level(Box<Level>),
}

/// What a level does with the tiles below it: spread them across hardware instances, or
/// partition them sequentially across a grid. Decided once, when the level is built, so no
/// consumer re-folds the per-axis distributions.
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
    role: LevelRole,
    order: WalkOrder,
    schedule: Schedule,
    next: Partitioner,
}

impl Level {
    pub fn schedule(&self) -> Schedule {
        self.schedule
    }

    pub(crate) fn role(&self) -> LevelRole {
        self.role
    }
}

impl Partitioner {
    pub fn is_final(&self) -> bool {
        matches!(self, Partitioner::Final(_))
    }

    /// The [`Leaf`] instruction at the end of the chain.
    pub fn leaf(&self) -> Leaf {
        match self {
            Partitioner::Final(leaf) => *leaf,
            Partitioner::Level(level) => level.next.leaf(),
        }
    }

    /// Set the chain-end [`Leaf`], after all levels are stacked (appending a level
    /// resets it to the tail's).
    pub(crate) fn with_leaf(self, leaf: Leaf) -> Partitioner {
        match self {
            Partitioner::Final(_) => Partitioner::Final(leaf),
            Partitioner::Level(mut level) => {
                level.next = level.next.with_leaf(leaf);
                Partitioner::Level(level)
            }
        }
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

    /// This level's [`LevelRole`]. Panics on [`Final`](Partitioner::Final), which carries no level.
    pub(crate) fn role(&self) -> LevelRole {
        self.level().role
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

    pub fn schedule(&self) -> Schedule {
        self.level().schedule
    }

    /// Resolve every level's deferred [`PlaneLanes`](super::Coverage::PlaneLanes) count to
    /// `Instances(plane_size)`. The launch's single stamping pass, so geometry and the walk
    /// only ever see concrete instance counts.
    pub(crate) fn resolve_lanes(self, plane_size: usize) -> Partitioner {
        match self {
            Partitioner::Final(leaf) => Partitioner::Final(leaf),
            Partitioner::Level(level) => {
                let Level {
                    edges,
                    dists,
                    role,
                    order,
                    schedule,
                    next,
                } = *level;
                // Resolving lane counts keeps every axis `Spatial`, so the role is unchanged.
                Partitioner::Level(Box::new(Level {
                    edges,
                    dists: dists.map(|_, d| d.resolve_lanes(plane_size)),
                    role,
                    order,
                    schedule,
                    next: next.resolve_lanes(plane_size),
                }))
            }
        }
    }

    pub(crate) fn append(self, tail: Partitioner) -> Partitioner {
        match self {
            Partitioner::Final(_) => tail,
            Partitioner::Level(level) => {
                let Level {
                    edges: sub_tile,
                    dists,
                    role,
                    order,
                    schedule,
                    next,
                } = *level;
                Partitioner::Level(Box::new(Level {
                    edges: sub_tile,
                    dists,
                    role,
                    order,
                    schedule,
                    next: next.append(tail),
                }))
            }
        }
    }

    /// Panics on [`Final`](Partitioner::Final), which carries no level.
    fn level(&self) -> &Level {
        match self {
            Partitioner::Level(level) => level,
            Partitioner::Final(_) => {
                panic!(
                    "Partitioner: the final partitioner carries no level (check `is_final` first)"
                )
            }
        }
    }
}

/// A [`Partitioner`] with its split and walk order set but no [`Schedule`] yet.
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
    fn finish(self, schedule: Schedule) -> Partitioner {
        // Instance when any axis spreads across hardware, else a sequential partition.
        let role = self
            .dists
            .values()
            .fold(LevelRole::Partition, |role, dist| match dist {
                Distribution::Spatial { .. } => LevelRole::Instance,
                Distribution::Sequential => role,
            });
        Partitioner::Level(Box::new(Level {
            edges: self.sub_tile,
            dists: self.dists,
            role,
            order: self.order,
            schedule,
            next: Partitioner::Final(Leaf::Register),
        }))
    }

    pub fn staged(self) -> Partitioner {
        self.finish(Schedule::Staged)
    }

    pub fn direct(self) -> Partitioner {
        self.finish(Schedule::Direct)
    }

    pub fn double_buffered(self) -> Partitioner {
        self.finish(Schedule::DoubleBuffered)
    }
}
