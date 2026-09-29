//! What a walk's slots hold ([`Payload`]).

use cubecl::prelude::*;

use crate::*;

/// One operand as a slot plans it.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StageOperand {
    /// Who moves this operand's bytes into its stage.
    pub(crate) delivery: Delivery,
    /// The axes the operand spans.
    pub(crate) space: Space,
}

/// What shapes a slot's buffers: level, depth, storage and optional line width.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StageSpec {
    pub(crate) level: Level,
    pub(crate) depth: usize,
    pub(crate) storage: StageStorage,
    pub(crate) width: Option<usize>,
}

/// One operand or two, staged and filled as one.
#[cube]
pub(crate) trait Payload<Me: CubeType>: CubeType {
    /// These operands as a slot plans them, in the order this payload holds them.
    fn operands(&self) -> comptime_type!(Vec<StageOperand>);

    /// A staged copy of these operands, reusing `first`'s buffer where `refills` marks it
    /// [`Shared`](Refill::Shared).
    fn staged(
        &self,
        first: &Me,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> Me;

    /// Bring `src`'s window at `region` into this payload, for each operand whose refill is `only`.
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
