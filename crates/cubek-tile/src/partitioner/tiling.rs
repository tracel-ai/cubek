//! A level-centric builder for a multi-level [`Space`]. Declare the axis extents once,
//! then one [`level`](LeveledTiling::level) per decomposition: its walk order, schedule,
//! and the per-axis [`Cut`]. Each [`level`](LeveledTiling::level) maps 1:1 to the
//! [`Level`](super::Level) the [`Walk`](crate::Walk) consumes — no transpose.

use crate::{Axis, ByAxis, Space};

use super::{CubeAxis, Distribution, Partitioner, Schedule, WalkOrder};

/// How one axis is cut at one level: the sub-tile `edge` and how that level hands the
/// tiles out. Constructors name the common distributions; [`Cut::new`] takes any.
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    pub edge: usize,
    pub dist: Distribution,
}

impl Cut {
    pub fn new(edge: usize, dist: Distribution) -> Self {
        Cut { edge, dist }
    }

    /// `edge`-sized tiles dealt one-per-cube along `axis`.
    pub fn cube(axis: CubeAxis, edge: usize) -> Self {
        Cut::new(edge, Distribution::cube(axis))
    }

    /// `edge`-sized tiles dealt one-per-plane (worker thread).
    pub fn plane(edge: usize) -> Self {
        Cut::new(edge, Distribution::plane())
    }

    /// `edge`-sized tiles walked sequentially by one instance.
    pub fn sequential(edge: usize) -> Self {
        Cut::new(edge, Distribution::Sequential)
    }
}

/// One decomposition level: its walk order, schedule, and the [`Cut`] for every axis.
struct LevelSpec {
    order: WalkOrder,
    schedule: Schedule,
    cuts: Vec<(Axis, Cut)>,
}

/// The empty seed: declare [`extents`](Tiling::extents) to start adding levels.
pub struct Tiling;

impl Tiling {
    pub fn new() -> Self {
        Tiling
    }

    /// Declare every axis and its top extent, fixing the canonical axis order. Levels are
    /// added next; their cuts may come in any order and are realigned to this one.
    pub fn extents(self, extents: &[(Axis, usize)]) -> LeveledTiling {
        LeveledTiling {
            extents: extents.to_vec(),
            levels: Vec::new(),
        }
    }
}

impl Default for Tiling {
    fn default() -> Self {
        Tiling::new()
    }
}

/// Builds a [`Space`] one level at a time. Add levels with [`level`](LeveledTiling::level),
/// each configured by a closure that hangs the per-axis [`Cut`]s off a [`LevelBuilder`],
/// then [`build`](LeveledTiling::build).
pub struct LeveledTiling {
    extents: Vec<(Axis, usize)>,
    levels: Vec<LevelSpec>,
}

impl LeveledTiling {
    /// Add a decomposition level (coarse to fine) with its walk order and schedule; `cuts`
    /// hangs the per-axis [`Cut`]s off the [`LevelBuilder`].
    pub fn level(
        mut self,
        order: WalkOrder,
        schedule: Schedule,
        cuts: impl FnOnce(LevelBuilder) -> LevelBuilder,
    ) -> Self {
        let level = cuts(LevelBuilder { cuts: Vec::new() });
        self.push(order, schedule, level.cuts);
        self
    }

    /// Close `cuts` into a level. They must cover exactly the declared axes (any order);
    /// [`build`](Self::build) realigns them to the extents' canonical order.
    fn push(&mut self, order: WalkOrder, schedule: Schedule, cuts: Vec<(Axis, Cut)>) {
        assert_eq!(
            cuts.len(),
            self.extents.len(),
            "LeveledTiling::level: {} cuts but {} axes declared",
            cuts.len(),
            self.extents.len()
        );
        for &(axis, _) in &self.extents {
            assert!(
                cuts.iter().any(|&(a, _)| a == axis),
                "LeveledTiling::level: axis {axis:?} has no cut"
            );
        }
        self.levels.push(LevelSpec {
            order,
            schedule,
            cuts,
        });
    }

    pub fn build(self) -> Space {
        let mut space = Space::new(&self.extents);
        for level in &self.levels {
            let cut = |axis| {
                level
                    .cuts
                    .iter()
                    .find(|&&(a, _)| a == axis)
                    .expect("checked in level()")
                    .1
            };
            // Realign each level's cuts to the canonical (extents) axis order.
            let edges: Vec<_> = self
                .extents
                .iter()
                .map(|&(a, _)| (a, cut(a).edge))
                .collect();
            let dists: Vec<_> = self
                .extents
                .iter()
                .map(|&(a, _)| (a, cut(a).dist))
                .collect();
            let builder = match level.order {
                WalkOrder::RowMajor => {
                    Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists))
                }
                WalkOrder::Reversed => {
                    Partitioner::reversed(ByAxis::new(&edges), ByAxis::new(&dists))
                }
            };
            let partitioner = match level.schedule {
                Schedule::Direct => builder.direct(),
                Schedule::Staged => builder.staged(),
                Schedule::DoubleBuffered => builder.double_buffered(),
            };
            space = space.with_partitioner(partitioner);
        }
        space
    }
}

/// Collects one level's per-axis [`Cut`]s, via [`axis`](Self::axis) for a single axis and
/// [`axes`](Self::axes) to hand a whole group the same cut.
pub struct LevelBuilder {
    cuts: Vec<(Axis, Cut)>,
}

impl LevelBuilder {
    /// One axis gets `cut`.
    pub fn axis(mut self, axis: Axis, cut: Cut) -> Self {
        self.cuts.push((axis, cut));
        self
    }

    /// Every axis in `axes` gets the same `cut` (e.g. all batch axes pinned alike).
    pub fn axes(mut self, axes: &[Axis], cut: Cut) -> Self {
        self.cuts.extend(axes.iter().map(|&a| (a, cut)));
        self
    }
}
