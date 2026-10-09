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

/// What shapes a slot's buffers: level, depth, storage, optional line width, and who holds them.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StageSpec {
    pub(crate) level: Level,
    pub(crate) depth: usize,
    pub(crate) storage: StageStorage,
    pub(crate) width: Option<usize>,
    pub(crate) owner: StageOwner,
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

    /// Free the shared memory this payload's stages hold, for what is declared after it to take.
    fn free(&self);

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

/// One operand's stage, placed in the operand's nest at the depth the walk's regions sit below.
/// The nest's levels come with it: a plane's window of the stage is the plane's, and a landing
/// out of it takes a window for every plane the levels lay out.
#[cube]
pub(crate) fn stage_one<T: Numeric>(operand: &Tile<T>, #[comptime] spec: StageSpec) -> Tile<T> {
    let stage = Memory::stage(
        operand,
        comptime!(spec.level.clone()),
        comptime!(spec.storage.clone()),
        comptime!(spec.width),
        comptime!(spec.owner),
    );
    let stage = Tile::new(
        stage.kind,
        comptime!(Placement::new(
            stage.place.space.clone(),
            spec.depth,
            operand.place.levels.clone()
        )),
    );
    stage.with_scales_staged(operand, comptime!(spec))
}
