//! The [`Partitioner`]: a recursive descent strategy for a [`Space`](crate::Space),
//! one decomposition level plus the partitioner for the subspaces it produces.

use crate::{Axis, ByAxis};

use super::{Distribution, WalkOrder};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Schedule {
    Direct,
    Staged,
    /// Staged with two buffers, prefetching the next sub-tile while computing.
    DoubleBuffered,
}

/// A space holds exactly one; [`divide`](crate::Space::divide) consumes the level and
/// hands [`next`](Partitioner::next) down.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Partitioner {
    Final,
    Level(Box<Level>),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    sub_tile: ByAxis<usize>,
    dists: ByAxis<Distribution>,
    order: WalkOrder,
    schedule: Schedule,
    next: Partitioner,
}

impl Partitioner {
    pub fn is_final(&self) -> bool {
        matches!(self, Partitioner::Final)
    }

    pub fn next(&self) -> &Partitioner {
        &self.level().next
    }

    pub fn edge(&self, axis: Axis) -> usize {
        self.level().sub_tile.get(axis)
    }

    pub fn distribution(&self, axis: Axis) -> Distribution {
        self.level().dists.get(axis)
    }

    pub fn order(&self) -> WalkOrder {
        self.level().order
    }

    pub fn schedule(&self) -> Schedule {
        self.level().schedule
    }

    pub(crate) fn append(self, tail: Partitioner) -> Partitioner {
        match self {
            Partitioner::Final => tail,
            Partitioner::Level(level) => {
                let Level {
                    sub_tile,
                    dists,
                    order,
                    schedule,
                    next,
                } = *level;
                Partitioner::Level(Box::new(Level {
                    sub_tile,
                    dists,
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
            Partitioner::Final => {
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
        Partitioner::Level(Box::new(Level {
            sub_tile: self.sub_tile,
            dists: self.dists,
            order: self.order,
            schedule,
            next: Partitioner::Final,
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
