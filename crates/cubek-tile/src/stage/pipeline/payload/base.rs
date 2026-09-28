//! What a walk's slots hold ([`Payload`]): the trait, what it reads off one operand
//! ([`StageOperand`]) and what the walk states about the buffers ([`StageSpec`]).

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
pub(crate) struct StageSpec {
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

    /// A staged copy of these operands, shaped by `spec`: each one a fresh buffer, or the
    /// buffer `first` already holds where `refills` marks it [`Shared`](Refill::Shared).
    ///
    /// `first` is the first slot's payload, which owns every buffer. At the first slot nothing is
    /// shared, so what is passed there is never read.
    fn staged(
        &self,
        first: &Me,
        #[comptime] spec: StageSpec,
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

/// One operand's stage, placed at the depth the walk's regions sit below.
///
/// A gathered operand keeps its compacted physical window and projection, so staging does not
/// replicate each logical element for every gather tap; the leaf performs the gather on read.
#[cube]
pub(crate) fn stage_one<T: Numeric>(operand: &Tile<T>, #[comptime] spec: StageSpec) -> Tile<T> {
    let stage = Memory::stage(
        operand,
        comptime!(spec.level),
        comptime!(spec.storage),
        comptime!(spec.width),
    );
    Tile::new(stage.kind, comptime!(stage.place.at_depth(spec.depth)))
}
