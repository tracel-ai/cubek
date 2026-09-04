//! A level-centric builder for a multi-level [`Space`]: [`Tiling::over`] declares the axes and
//! their extents, then one [`level`](Tiling::level) per decomposition states who works on which
//! axes ([`LevelCuts`]), coarse to fine. Each level maps 1:1 to the [`Level`](super::Level) the
//! [`Walk`](crate::Walk) consumes; no transpose.
//!
//! Geometry only. How a level is walked (its order, how deep its stages are buffered), where an
//! operand is materialized and what runs on the cells are the kernel's to write, level by level,
//! against this space.

use crate::{Axis, ByAxis, Extent, Space};

use super::{ComputeScope, Distribution, Handout, Partitioner, Spatial, Spread, Work};

/// How one axis is cut at one level: the sub-tile `edge` and how that level hands the tiles out.
///
/// What a level ends up holding, not what a caller writes: the two verbs on [`LevelCuts`] build
/// these, and nothing outside this module states one.
#[derive(Clone, Copy, Debug)]
struct Cut {
    edge: usize,
    dist: Distribution,
}

impl Cut {
    fn new(edge: usize, dist: Distribution) -> Self {
        Cut { edge, dist }
    }

    /// `edge`-sized tiles walked by every worker the level runs on.
    fn sequential(edge: usize) -> Self {
        Cut::new(edge, Distribution::Sequential)
    }
}

/// One decomposition level: the [`Cut`] for every axis, and the axes it distributes as one.
struct LevelSpec {
    /// The axes this level distributes as one, if any ([`LevelCuts::distribute`]).
    work: Option<Work>,
    cuts: Vec<(Axis, Cut)>,
}

impl LevelSpec {
    fn cut(&self, axis: Axis) -> Cut {
        self.cuts
            .iter()
            .find(|&&(a, _)| a == axis)
            .expect("checked in level()")
            .1
    }
}

/// Builds a [`Space`] one level at a time. Start with [`over`](Tiling::over) (static extents)
/// or [`axes`](Tiling::axes) (extents resolved in-kernel), add levels with
/// [`level`](Tiling::level), then [`build`](Tiling::build).
pub struct Tiling {
    extents: Vec<(Axis, Extent)>,
    levels: Vec<LevelSpec>,
}

impl Tiling {
    /// Declare every axis and its top extent, fixing the canonical axis order; cuts may come in
    /// any order and are realigned to it.
    pub fn over(extents: &[(Axis, usize)]) -> Tiling {
        Tiling {
            extents: extents
                .iter()
                .map(|&(axis, extent)| (axis, Extent::Static(extent)))
                .collect(),
            levels: Vec::new(),
        }
    }

    /// [`over`](Tiling::over) with every top extent [`Dynamic`](Extent::Dynamic): the kernel
    /// form, resolved in-kernel from the tensors, so one compiled kernel serves every shape. The
    /// launch stamps the real extents back on with [`Space::with_extents`].
    pub fn axes(axes: &[Axis]) -> Tiling {
        Tiling {
            extents: axes.iter().map(|&axis| (axis, Extent::Dynamic)).collect(),
            levels: Vec::new(),
        }
    }

    /// Add a decomposition level (coarse to fine): `f` states who works on which axes, on the
    /// collector. A level that cuts nothing is one region, and is kept: the kernel walks what
    /// was stated, level for level.
    pub fn level(mut self, f: impl FnOnce(&mut LevelCuts)) -> Self {
        let mut cuts = LevelCuts::new();
        f(&mut cuts);
        self.push(cuts.cuts, cuts.work);
        self
    }

    /// Build the [`Space`]: the extents and the stack of levels.
    pub fn build(self) -> Space {
        let mut space = Space::from_extents(&self.extents);
        for level in &self.levels {
            let edges = self.edges(level);
            let dists: Vec<_> = self
                .extents
                .iter()
                .map(|&(a, _)| (a, level.cut(a).dist))
                .collect();
            space = space.with_partitioner(
                Partitioner::over(ByAxis::new(&edges), ByAxis::new(&dists))
                    .distributing(level.work.clone()),
            );
        }
        space
    }

    /// Close `cuts` into a level. They must cover exactly the declared axes (any order);
    /// [`build`](Self::build) realigns them to the extents' canonical order.
    fn push(&mut self, cuts: Vec<(Axis, Cut)>, work: Option<Work>) {
        // Per axis first: it names the one that is wrong, where the count only says the total
        // is off.
        for &(axis, _) in &self.extents {
            let stated = cuts.iter().filter(|&&(a, _)| a == axis).count();
            assert!(stated > 0, "Tiling::level: axis {axis:?} has no cut");
            assert!(
                stated == 1,
                "Tiling::level: axis {axis:?} is cut {stated} times; a level states each \
                 of its axes once, by `walk` or `distribute`"
            );
        }
        assert_eq!(
            cuts.len(),
            self.extents.len(),
            "Tiling::level: {} cuts but {} axes declared",
            cuts.len(),
            self.extents.len()
        );
        self.levels.push(LevelSpec { work, cuts });
    }

    /// One level's sub-tile edges, realigned to the canonical (extents) axis order.
    fn edges(&self, level: &LevelSpec) -> Vec<(Axis, usize)> {
        self.extents
            .iter()
            .map(|&(a, _)| (a, level.cut(a).edge))
            .collect()
    }
}

/// Collects what one level says, in two verbs: [`distribute`](Self::distribute) hands some axes'
/// regions to a hardware scope's workers, [`walk`](Self::walk) leaves the rest to every one of
/// them. Between them they name each of the space's axes exactly once.
pub struct LevelCuts {
    cuts: Vec<(Axis, Cut)>,
    work: Option<Work>,
}

impl LevelCuts {
    fn new() -> Self {
        LevelCuts {
            cuts: Vec::new(),
            work: None,
        }
    }

    /// Nobody owns these axes: every worker on this level covers all of their tiles, walking them
    /// one at a time. `axes` pairs each axis with its tile edge.
    ///
    /// The contraction of a matmul is the everyday one. So is any axis a level leaves alone,
    /// which is most of them at most levels.
    pub fn walk(&mut self, axes: &[(Axis, usize)]) -> &mut Self {
        self.cuts.extend(
            axes.iter()
                .map(|&(axis, edge)| (axis, Cut::sequential(edge))),
        );
        self
    }

    /// Hand these axes' regions to `dist`'s workers, in order: as many workers as regions means
    /// one each, fewer means each takes a run. `axes` pairs each axis with its tile edge, and
    /// `dist` says who runs the tiles and how many of them there are ([`cubes`](crate::cubes),
    /// [`planes`](crate::planes), [`lanes`](crate::lanes) and their knobs).
    ///
    /// One axis, or one region each, is a box of the grid, so every axis gets a dial of its own
    /// and two lines make a two-dimensional box. Several axes sharing a stated count are read as a
    /// single index instead, so a worker takes a share of the whole rather than a box of it: a
    /// dial per axis always yields a box, and a share that begins inside one region and ends
    /// inside another is not one.
    ///
    /// That index runs over these axes' tiles at this level and one region of the level below, so
    /// a share can end part way through a region's own work ([`Walk::window`](crate::Walk::window)
    /// is how a kernel takes its share).
    ///
    /// An axis named here is not named by [`walk`](Self::walk): a level states each of its axes
    /// once.
    pub fn distribute(&mut self, dist: Spatial, axes: &[(Axis, usize)]) -> &mut Self {
        match dist.handout(axes.len()) {
            Handout::Dial => self.cuts.extend(
                axes.iter()
                    .map(|&(axis, edge)| (axis, Cut::new(edge, dist.into()))),
            ),
            Handout::OneIndex => {
                assert!(
                    self.work.is_none(),
                    "LevelCuts::distribute: this level already distributes work; state it once"
                );
                // A share is a range of one index, and lanes that reduce in registers have to
                // reach their reduction together. Ranges put them on different regions, so they
                // never would.
                assert!(
                    dist.scope() != ComputeScope::Unit,
                    "LevelCuts::distribute: the plane's lanes combine in registers, which needs \
                     them in lockstep, and lanes holding different shares never are. Distribute \
                     one axis at a time across them instead."
                );
                // A share is walked as a nest: consecutive steps of one region under one
                // accumulator, then the next region. Turns would put a different region under it
                // at every step.
                assert!(
                    dist.spread() == Spread::Contiguous,
                    "LevelCuts::distribute: a share is a run of the index, so its steps are \
                     consecutive; instances taking turns would leave no region long enough to \
                     accumulate in. Distribute one interleaved axis instead."
                );
                self.cuts.extend(
                    axes.iter()
                        .map(|&(axis, edge)| (axis, Cut::sequential(edge))),
                );
                self.work = Some(Work::new(
                    axes.iter().map(|&(axis, _)| axis).collect(),
                    dist,
                ));
            }
        }
        self
    }
}
