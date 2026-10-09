//! How an accumulator's tiles reach memory: the plan ([`DrainPlan`]), the pass at each leaf
//! ([`DrainPass`]) and which units carry the writes ([`Drain`]).

use crate::*;

/// How a plane-resident accumulator's tiles reach memory, per its [`Scratch`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DrainPlan {
    /// No scratch: each tile stores through its own intrinsic.
    Straight,
    /// Own slot per tile: all spills, then all adds, two barriers in all.
    BounceTogether,
    /// One shared slot: spill and add per tile, three barriers each.
    BounceEach,
}

impl DrainPlan {
    pub(crate) fn new(scratch: Scratch) -> Self {
        match scratch {
            Scratch::None => DrainPlan::Straight,
            Scratch::OneTile => DrainPlan::BounceEach,
            Scratch::WholeGrid => DrainPlan::BounceTogether,
        }
    }
}

/// What a drain does to one tile at the leaf of its descent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum DrainPass {
    /// Store through the tile's own intrinsic.
    Copy,
    /// Spill, add, and the three barriers around them.
    Bounce,
    /// Spill alone: the first pass.
    Spill,
    /// Add alone: the second pass.
    Add,
}

/// Which of the plane's units carry a drain's writes, and what they combine first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Drain {
    /// Each unit holds whole cells of its own and writes them.
    EachUnit,
    /// Every unit holds the same whole cells and the write folds, so one unit writes.
    UnitZero,
    /// Units each hold a partial of one cell: combine across the plane, unit zero writes.
    PlaneFold,
    /// Groups each hold a partial of one cell: combine within the group, its first unit writes.
    GroupFold { unit_bits: usize },
}

impl Drain {
    pub(crate) const fn of(units: UnitShare, write: Write) -> Self {
        match (units, write) {
            (UnitShare::Plane, _) => Drain::PlaneFold,
            (UnitShare::Group { unit_bits }, _) => Drain::GroupFold { unit_bits },
            // Repeated units hold the same cells: a store may land many times, a fold only once.
            (UnitShare::Repeated, Write::Accumulate | Write::Exclusive(_)) => Drain::UnitZero,
            (UnitShare::Repeated, Write::Replace) | (UnitShare::Whole, _) => Drain::EachUnit,
        }
    }
}
