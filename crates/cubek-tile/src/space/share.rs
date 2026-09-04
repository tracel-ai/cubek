//! What the hardware instances are to a tile's cells, once a level has been dealt out.
//!
//! Two questions, at two scopes. A plane's lanes share registers, so they combine there and a
//! folding drain must know *which* lanes hold a cell ([`LaneShare`]) and how many run the work
//! ([`LaneWork`]). Planes and cubes share none, so each folds its own contribution into the
//! destination and the answer is only whether it holds a whole cell ([`SplitShare`]).
//!
//! The vocabulary and the [`Space`] descent that derives it, together: the enums are only ever
//! read off a space, and the descent is only ever read as one of them.



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

/// A descent's split share, given the parent's and the level's: partial stays partial.
pub(crate) fn join_split_share(parent: SplitShare, level: SplitShare) -> SplitShare {
    match (parent, level) {
        (SplitShare::Whole, SplitShare::Whole) => SplitShare::Whole,
        (SplitShare::Partial, _) | (_, SplitShare::Partial) => SplitShare::Partial,
    }
}

/// A descent's lane work, given the parent's and whether the level rides lanes: once something
/// does, every lane below has its own share.
pub(crate) fn join_lane_work(parent: LaneWork, rides: bool) -> LaneWork {
    match (parent, rides) {
        (LaneWork::Own, _) | (_, true) => LaneWork::Own,
        (LaneWork::Repeated, false) => LaneWork::Repeated,
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
                 (`distribute(lanes(n), ..)`, combined in the plane's registers), or give the \
                 output an axis of its own for the split."
            ),
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
