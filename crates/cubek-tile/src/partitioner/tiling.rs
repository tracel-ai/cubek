//! An axis-centric builder for a multi-level [`Space`]. Declare each axis once with its
//! coarse→fine `(edge, distribution)` per level; the builder transposes that into the
//! leveled [`Partitioner`] the [`Walk`](crate::Walk) consumes.

use crate::{Axis, ByAxis, Space};

use super::{CubeAxis, Distribution, Partitioner, Schedule, WalkOrder};

/// How one axis is split at one level: the sub-tile `edge` and how that level hands the
/// tiles out. Constructors name the common distributions; [`Split::new`] takes any.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub edge: usize,
    pub dist: Distribution,
}

impl Split {
    pub fn new(edge: usize, dist: Distribution) -> Self {
        Split { edge, dist }
    }

    /// `edge`-sized tiles dealt one-per-cube along `axis`.
    pub fn cube(axis: CubeAxis, edge: usize) -> Self {
        Split::new(edge, Distribution::cube(axis))
    }

    /// `edge`-sized tiles dealt one-per-plane (worker thread).
    pub fn plane(edge: usize) -> Self {
        Split::new(edge, Distribution::plane())
    }

    /// `edge`-sized tiles walked sequentially by one instance.
    pub fn seq(edge: usize) -> Self {
        Split::new(edge, Distribution::Sequential)
    }
}

/// Builds a [`Space`] level by level, but described per axis. Each [`axis`](Tiling::axis)
/// call appends one axis (in declared order) plus its split at every level; [`build`](Tiling::build)
/// transposes those into one [`Partitioner`] level per schedule.
pub struct Tiling {
    order: WalkOrder,
    schedules: Vec<Schedule>,
    extents: Vec<(Axis, usize)>,
    /// `levels[l]` is the `(axis, edge, distribution)` of every axis at level `l`.
    levels: Vec<Vec<(Axis, usize, Distribution)>>,
}

impl Tiling {
    /// One level per [`Schedule`], all walked row-major.
    pub fn row_major(schedules: &[Schedule]) -> Self {
        Self::new(WalkOrder::RowMajor, schedules)
    }

    pub fn new(order: WalkOrder, schedules: &[Schedule]) -> Self {
        Tiling {
            order,
            schedules: schedules.to_vec(),
            extents: Vec::new(),
            levels: vec![Vec::new(); schedules.len()],
        }
    }

    /// Declare `axis` (top `extent`) and its [`Split`] at each level, coarse to fine.
    /// `splits.len()` must equal the level count.
    pub fn axis(mut self, axis: Axis, extent: usize, splits: &[Split]) -> Self {
        assert_eq!(
            splits.len(),
            self.schedules.len(),
            "Tiling::axis: axis {axis:?} has {} splits but there are {} levels",
            splits.len(),
            self.schedules.len()
        );
        self.extents.push((axis, extent));
        for (level, split) in self.levels.iter_mut().zip(splits) {
            level.push((axis, split.edge, split.dist));
        }
        self
    }

    pub fn build(self) -> Space {
        let mut space = Space::new(&self.extents);
        for (level, &schedule) in self.levels.iter().zip(&self.schedules) {
            let edges: Vec<_> = level.iter().map(|&(a, e, _)| (a, e)).collect();
            let dists: Vec<_> = level.iter().map(|&(a, _, d)| (a, d)).collect();
            let builder = match self.order {
                WalkOrder::RowMajor => {
                    Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists))
                }
                WalkOrder::Reversed => {
                    Partitioner::reversed(ByAxis::new(&edges), ByAxis::new(&dists))
                }
            };
            let partitioner = match schedule {
                Schedule::Direct => builder.direct(),
                Schedule::Staged => builder.staged(),
                Schedule::DoubleBuffered => builder.double_buffered(),
            };
            space = space.with_partitioner(partitioner);
        }
        space
    }
}
