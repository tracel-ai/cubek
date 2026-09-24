//! What a walk's slots hold ([`Payload`]): one operand, or two.
//!
//! Written once for either shape, so the constructors, the plan and the fill below are the same
//! code whatever a walk stages. A payload only counts its operands and hands each one over; an
//! operation names its own roles.

use cubecl::prelude::*;

use crate::*;

/// What one operand is to the slot planning it: who moves its bytes, and the axes it spans.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StageOperand {
    /// Who moves this operand's bytes into its stage ([`Tile::delivery`]).
    pub(crate) delivery: Delivery,
    /// The axes the operand spans, which decide whether a walk moves its window.
    pub(crate) space: Space,
}

/// What a slot's buffers are shaped by: the level one region of the walk sits at, the depth the
/// stage is placed at, how it lays its cells out, and the line it is served in where the caller
/// states one rather than taking the operand's.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Staging {
    pub(crate) level: Level,
    pub(crate) depth: usize,
    pub(crate) storage: StageStorage,
    pub(crate) width: Option<usize>,
}

/// One operand or two, staged and filled as one.
///
/// **The arity is the payload's**, so everything above it — the plan, the slots, the schedule —
/// is written once. A shape implements the three things a slot does to its operands: read their
/// comptime facts, stage them, and bring a region's window into the stage.
#[cube]
pub(crate) trait Payload<Me: CubeType>: CubeType {
    /// These operands as a slot plans them, in the order this payload holds them.
    fn operands(&self) -> comptime_type!(Vec<StageOperand>);

    /// A staged copy of these operands, shaped by `staging`: each one a fresh buffer, or the
    /// buffer `first` already holds where `refills` marks it [`Shared`](Refill::Shared).
    ///
    /// `first` is the first slot's payload, which owns every buffer. At the first slot nothing is
    /// shared, so what is passed there is never read.
    fn staged(
        &self,
        first: &Me,
        #[comptime] staging: Staging,
        #[comptime] refills: Vec<Refill>,
    ) -> Me;

    /// Bring `src`'s window at `region` into this staged payload, for each operand whose refill
    /// is `only` and no other. `meeting` moves the bytes, the delivery deciding how.
    fn bring(
        &mut self,
        src: &Me,
        meeting: &Meeting,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
        #[comptime] only: Refill,
    );
}

#[cube]
impl<T: Numeric> Payload<Tile<T>> for Tile<T> {
    fn operands(&self) -> comptime_type!(Vec<StageOperand>) {
        let delivery = self.delivery();
        comptime!(vec![StageOperand {
            delivery,
            space: self.place.space.clone(),
        }])
    }

    fn staged(
        &self,
        first: &Tile<T>,
        #[comptime] staging: Staging,
        #[comptime] refills: Vec<Refill>,
    ) -> Tile<T> {
        if comptime!(refills[0] == Refill::Shared) {
            first.clone()
        } else {
            stage_one(self, comptime!(staging))
        }
    }

    fn bring(
        &mut self,
        src: &Tile<T>,
        meeting: &Meeting,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
        #[comptime] only: Refill,
    ) {
        if comptime!(refills[0] == only) {
            meeting.fill(self, &src.at(region));
        }
    }
}

/// Two operands staged together: what a contraction's slots hold.
///
/// A named pair rather than a tuple, because a `#[cube]` trait is implemented for a named type.
/// `Clone` duplicates the handles, not the buffers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Pair<Lhs: Numeric, Rhs: Numeric> {
    pub lhs: Tile<Lhs>,
    pub rhs: Tile<Rhs>,
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Payload<Pair<Lhs, Rhs>> for Pair<Lhs, Rhs> {
    fn operands(&self) -> comptime_type!(Vec<StageOperand>) {
        let lhs = self.lhs.delivery();
        let rhs = self.rhs.delivery();
        comptime!(vec![
            StageOperand {
                delivery: lhs,
                space: self.lhs.place.space.clone(),
            },
            StageOperand {
                delivery: rhs,
                space: self.rhs.place.space.clone(),
            },
        ])
    }

    fn staged(
        &self,
        first: &Pair<Lhs, Rhs>,
        #[comptime] staging: Staging,
        #[comptime] refills: Vec<Refill>,
    ) -> Pair<Lhs, Rhs> {
        let lhs = if comptime!(refills[0] == Refill::Shared) {
            first.lhs.clone()
        } else {
            stage_one(&self.lhs, comptime!(staging.clone()))
        };
        let rhs = if comptime!(refills[1] == Refill::Shared) {
            first.rhs.clone()
        } else {
            stage_one(&self.rhs, comptime!(staging))
        };
        Pair::<Lhs, Rhs> { lhs, rhs }
    }

    fn bring(
        &mut self,
        src: &Pair<Lhs, Rhs>,
        meeting: &Meeting,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
        #[comptime] only: Refill,
    ) {
        if comptime!(refills[0] == only) {
            meeting.fill(&mut self.lhs, &src.lhs.at(region));
        }
        if comptime!(refills[1] == only) {
            meeting.fill(&mut self.rhs, &src.rhs.at(region));
        }
    }
}

/// One operand's stage, placed at the depth the walk's regions sit below.
///
/// A gathered operand keeps its compacted physical window and projection, so staging does not
/// replicate each logical element for every gather tap; the leaf performs the gather on read.
#[cube]
fn stage_one<T: Numeric>(operand: &Tile<T>, #[comptime] staging: Staging) -> Tile<T> {
    let stage = Memory::stage(
        operand,
        comptime!(staging.level),
        comptime!(staging.storage),
        comptime!(staging.width),
    );
    Tile::new(stage.kind, comptime!(stage.place.at_depth(staging.depth)))
}
