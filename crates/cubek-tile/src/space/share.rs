//! What the hardware instances are to a tile's cells, once a level has been dealt out.
//!
//! Two questions, at two scopes. A plane's lanes share registers, so they combine there and a
//! folding drain must know *which* lanes hold a cell ([`LaneShare`]) and how many run the work
//! ([`LaneWork`]). Planes and cubes share none, so each folds its own contribution into the
//! destination and the answer is only whether it holds a whole cell ([`SplitShare`]).
//!
//! The vocabulary and the [`Space`] descent that derives it, together: the enums are only ever
//! read off a space, and the descent is only ever read as one of them.

use crate::{Axis, ComputeScope, Distribution, Extent, Space};

/// What the plane's lanes each hold of a tile's cells, once a `Unit` split is dealt out. An axis
/// the tile doesn't span is *folded* (lanes cover disjoint slices, each holds a partial); one it
/// does span is *carried* (each lane gets a different cell). Which case a tile is in says how a
/// partial drains.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneShare {
    /// Nothing folded: the lane's cells are whole, so they read and write as they are.
    Whole,
    /// Nothing carried either, so every lane of the plane holds a partial of the *same* cell and
    /// the plane's own reduction is the drain.
    Plane,
    /// Both, so the plane splits into groups: one cell each, several cells in flight at once.
    /// `fold_mask` is the set of lane-index bits the folded axes occupy, so a cell's partials
    /// live on exactly the lanes that agree outside it and differ inside.
    Group { fold_mask: usize },
}

/// A descent's share, given the parent's and the level's: the folds compose, since each level
/// takes its own bits of the lane index. [`LaneShare::Plane`] already spans every lane, so nothing
/// folds under it, and nothing builds that: [`Space::cube_dim`](crate::Space::cube_dim) caps the
/// tree's `Unit` instance product at the plane width.
pub(crate) fn join_lane_share(parent: LaneShare, level: LaneShare) -> LaneShare {
    match (parent, level) {
        (LaneShare::Whole, share) | (share, LaneShare::Whole) => share,
        (LaneShare::Group { fold_mask: a }, LaneShare::Group { fold_mask: b }) => {
            LaneShare::Group { fold_mask: a | b }
        }
        _ => panic!("join_lane_share: {parent:?} under {level:?}: nothing folds under a plane"),
    }
}

/// How many of the plane's lanes run one tile's work. A space distributing nothing at `Unit` scope
/// still launches a full plane, every lane running the same code over the same cells. Identical
/// stores land the same value however many lanes make them, but a fold is not idempotent, so a
/// folding drain elects one lane. Distinct from [`LaneShare`], which says what a lane holds of a
/// cell rather than how many lanes hold it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneWork {
    /// Something rides the lanes, so each has its own share and a cell is written once.
    Own,
    /// Nothing does, so every lane repeats the same work and a cell is written once per lane.
    Repeated,
}

/// What the plane's lanes are to a tile's cells: what each holds of one ([`LaneShare`]), and how
/// many hold it ([`LaneWork`]). Two answers to one question, derived from the same space and read
/// together on drain, where neither settles who writes on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Lanes {
    pub share: LaneShare,
    pub work: LaneWork,
}

/// What one instance holds of a tile's cells, across the scopes whose instances can only meet in
/// the destination: `Plane` and `Cube`. [`LaneShare`]'s counterpart, and deliberately coarser: a
/// plane's lanes share registers and must elect a writer, hence a mask, but planes and cubes share
/// none, so each folds its own contribution and there is nothing to elect between them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SplitShare {
    /// Every cell this instance writes is its own outright, so the drain is a store.
    Whole,
    /// Several instances hold partials of the same cell, so the drain has to fold rather than
    /// store. A contraction cut at plane or cube scope is the way to get here.
    Partial,
}

impl SplitShare {
    /// Refuse an accumulation this share leaves in pieces, unless the destination adds them
    /// together. Called where an accumulator is opened and where one is written, the two places a
    /// partial can escape. A destination that replaces is wrong twice over and silently: a
    /// register drain stores, so the last instance erases the rest, and one accumulating in place
    /// loses the update. [`Write::Accumulate`](crate::Write) is the case this lets through.
    pub(crate) fn validate(self, write: crate::Write, site: &str) {
        match (self, write) {
            (SplitShare::Whole, _) | (SplitShare::Partial, crate::Write::Accumulate) => {}
            (SplitShare::Partial, crate::Write::Replace) => panic!(
                "{site}: this accumulator's cells are split across planes or cubes and its \
                 destination replaces rather than accumulates, so every partial but one would be \
                 lost. \
                 A contracted axis distributed across planes or cubes gives each instance a \
                 slice of the contraction, and none of them holds a whole cell. \
                 Drain into an accumulating destination (bind it as an `AccumulateArg`), \
                 distribute the contraction across the plane's lanes instead \
                 (`distribute(lanes(), ..)`, combined in the plane's registers), or give the \
                 output an axis of its own for the split."
            ),
        }
    }
}

impl Space {
    /// The share a tile ends up with at the leaf: every level's own share joined, the way
    /// [`MemData::at`](crate::MemData) joins them one at a time on the way down. A block built
    /// before the walk descends cannot read the stamped value, but every level is known here, so
    /// it can compute the value stamping would arrive at.
    pub(crate) fn leaf_lane_share(&self) -> LaneShare {
        let mut share = LaneShare::Whole;
        let mut level = self.clone();
        while !level.is_final() {
            share = join_lane_share(share, level.lane_share());
            level = level.divide();
        }
        share
    }

    /// What the plane's lanes are to this space's cells, both halves at once. What a drain is
    /// built from: [`leaf_lane_share`](Self::leaf_lane_share) alone cannot say who writes.
    pub(crate) fn lanes(&self) -> Lanes {
        Lanes {
            share: self.leaf_lane_share(),
            work: self.lane_work(),
        }
    }

    /// Whether anything rides this space's lanes, across every level. Read off the axes rather
    /// than off [`cube_dim`](Space::cube_dim), which asks the client for the hardware `plane_size`
    /// and is a launch-side question; this one is comptime, which is what a drain needs before it
    /// elects a writer.
    pub(crate) fn lane_work(&self) -> LaneWork {
        let mut level = self.clone();
        while !level.is_final() {
            let rides = level.partitioner().axes().into_iter().any(|axis| {
                let Distribution::Spatial {
                    scope: ComputeScope::Unit,
                    coverage,
                    ..
                } = level.partitioner().distribution(axis)
                else {
                    return false;
                };
                coverage.instances_const() != Some(1)
            });
            if rides {
                return LaneWork::Own;
            }
            level = level.divide();
        }
        LaneWork::Repeated
    }

    /// What one instance of `axes`' operand holds of its cells: [`Partial`](SplitShare::Partial)
    /// where a `Plane` or `Cube` axis the operand does not span is dealt across several instances,
    /// so each contracts a slice. Asked of the *whole* space, not the operand's projection: a
    /// projection has dropped the contracted axis and so cannot tell a split from a cut whose edge
    /// is the whole axis. Answered conservatively where the instance count is not comptime, since
    /// calling it whole loses every partial but one.
    pub(crate) fn split_share_of(&self, axes: &[Axis]) -> SplitShare {
        let mut level = self.clone();
        while !level.is_final() {
            // Work distributed as one is not an axis: a share of it covers part of a cell
            // whenever the index runs over an axis the operand does not span, and which part is
            // not something the level's per-axis distributions record.
            if let Some(work) = level.partitioner().work()
                && work.axes().iter().any(|axis| !axes.contains(axis))
            {
                return SplitShare::Partial;
            }
            let split = level.partitioner().axes().into_iter().any(|axis| {
                // An axis the operand spans is carried, not split: it gives each instance a cell
                // of its own rather than a slice of one.
                if axes.contains(&axis) {
                    return false;
                }
                let dist = level.partitioner().distribution(axis);
                match dist.scope() {
                    Some(ComputeScope::Cube(_)) | Some(ComputeScope::Plane) => {}
                    Some(ComputeScope::Unit) | None => return false,
                }
                level.instances_along(axis) != Some(1)
            });
            if split {
                return SplitShare::Partial;
            }
            level = level.divide();
        }
        SplitShare::Whole
    }

    /// How many instances `axis` is dealt out to at this level, where that is comptime: the pinned
    /// count, or the tile grid divided by each instance's share. `None` where the extent is
    /// [`Dynamic`](Extent::Dynamic) and so the grid is only known at runtime.
    fn instances_along(&self, axis: Axis) -> Option<usize> {
        let coverage = self.partitioner().distribution(axis).coverage();
        match coverage.instances_const() {
            Some(instances) => Some(instances),
            // `TilesEach`: the grid decides, and the grid needs the extent.
            None => match self.extent_raw(axis) {
                Extent::Static(_) => Some(coverage.instances(self.count(axis))),
                Extent::Dynamic => None,
            },
        }
    }

    /// The instance-index weight this space's own axis list cannot see: the instance counts of the
    /// same-scope axes *inside* `axis` that the partitioner distributes and this space does not
    /// span. A projected space is why: the index's odometer belongs to the partitioner, so an
    /// operand not spanning a contracted axis must still divide it out to find its own digit, and
    /// reading omitted axes as weight `1` aliases the outer digits onto one value. Panics where
    /// such an axis has no comptime count: assuming `1` is exactly that aliasing.
    pub(crate) fn inner_weight_unspanned(&self, axis: Axis) -> usize {
        if self.partitioner().is_final() {
            return 1;
        }
        let scope = self.partitioner().distribution(axis).scope();
        self.partitioner()
            .axes()
            .iter()
            .skip_while(|&&a| a != axis)
            .skip(1)
            .filter(|&&a| !self.contains(a) && self.partitioner().distribution(a).scope() == scope)
            .map(|&a| {
                self.partitioner()
                    .distribution(a)
                    .coverage()
                    .instances_const()
                    .unwrap_or_else(|| panic!(
                        "Space::inner_weight_unspanned: {a:?} is distributed inside {axis:?} at the \
                         same scope but this space does not span it, and its instance count is not \
                         comptime, so {axis:?}'s digit of the instance index cannot be decoded"
                    ))
            })
            .product()
    }

    pub(crate) fn lane_share(&self) -> LaneShare {
        if self.partitioner().is_final() {
            return LaneShare::Whole;
        }
        // Innermost first, so `weight` is the axis's stride in the lane index as it is reached,
        // the same least-significant-last ordering `Walk::from_counts` decodes with.
        let (mut weight, mut fold_mask) = (1usize, 0usize);
        for axis in self.partitioner().axes().into_iter().rev() {
            let Distribution::Spatial {
                scope: ComputeScope::Unit,
                coverage,
                ..
            } = self.partitioner().distribution(axis)
            else {
                continue;
            };
            // Asserted, not skipped: a `Unit` axis always resolves to `Instances`
            // (`Distribution::unit` defers through `PlaneLanes`), and passing over one whose
            // count we could not read would shift every inner axis's bits by its width.
            let lanes = coverage
                .instances_const()
                .expect("Space::lane_share: a Unit axis must carry a const instance count");
            if lanes == 1 {
                continue;
            }
            assert!(
                lanes.is_power_of_two(),
                "Space::lane_share: {axis:?} rides {lanes} lanes, which is not a power of two, so its partials are not a bit range"
            );
            if !self.contains(axis) {
                fold_mask |= (lanes - 1) * weight;
            }
            weight *= lanes;
        }
        match fold_mask {
            0 => LaneShare::Whole,
            // Every lane's bit folded: nothing is carried, so the plane shares the one cell.
            mask if mask == weight - 1 => LaneShare::Plane,
            fold_mask => LaneShare::Group { fold_mask },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Write;

    /// A destination that replaces cannot take a cell several instances hold slices of. The
    /// guard is the only thing between that mistake and a wrong number: nothing about it shows up
    /// at compile time or in a crash.
    #[test]
    #[should_panic(expected = "split across planes or cubes")]
    fn a_partial_cell_may_not_be_stored() {
        SplitShare::Partial.validate(Write::Replace, "test");
    }

    /// A destination that accumulates is exactly the case it exists to let through.
    #[test]
    fn a_partial_cell_may_be_accumulated() {
        SplitShare::Partial.validate(Write::Accumulate, "test");
    }

    /// A whole cell is nobody else's business, whichever way it is written.
    #[test]
    fn a_whole_cell_is_written_either_way() {
        SplitShare::Whole.validate(Write::Replace, "test");
        SplitShare::Whole.validate(Write::Accumulate, "test");
    }
}
