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

/// The instruction that contracts a final tile: the software microkernel over the
/// accumulator's memory, or the tensor-core path, which hoists the accumulator into a
/// plane-resident fragment at the recursion boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Leaf {
    #[default]
    Register,
    Cmma,
}

/// A space holds exactly one; [`divide`](crate::Space::divide) consumes the level and
/// hands [`next`](Partitioner::next) down. `Final` carries the [`Leaf`] instruction the
/// innermost tile contracts with.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Partitioner {
    Final(Leaf),
    Level(Box<Level>),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    edges: ByAxis<usize>,
    dists: ByAxis<Distribution>,
    order: WalkOrder,
    schedule: Schedule,
    next: Partitioner,
}

impl Level {
    pub fn schedule(&self) -> Schedule {
        self.schedule
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
    pub fn with_leaf(self, leaf: Leaf) -> Partitioner {
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

    pub fn order(&self) -> WalkOrder {
        self.level().order
    }

    pub fn schedule(&self) -> Schedule {
        self.level().schedule
    }

    pub(crate) fn append(self, tail: Partitioner) -> Partitioner {
        match self {
            Partitioner::Final(_) => tail,
            Partitioner::Level(level) => {
                let Level {
                    edges: sub_tile,
                    dists,
                    order,
                    schedule,
                    next,
                } = *level;
                Partitioner::Level(Box::new(Level {
                    edges: sub_tile,
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
        Partitioner::Level(Box::new(Level {
            edges: self.sub_tile,
            dists: self.dists,
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
