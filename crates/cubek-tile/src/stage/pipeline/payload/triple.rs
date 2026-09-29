//! The three-operand payload: what a walk holding two contractions' operands stages.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

/// Three operands staged together; `Clone` duplicates handles, not buffers.
///
/// An attention's walk stages its query, fixed across the walk and filled once, beside the keys
/// and values each step reads.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct OperandTriple<A: Numeric, B: Numeric, C: Numeric> {
    pub a: Tile<A>,
    pub b: Tile<B>,
    pub c: Tile<C>,
}

#[cube]
impl<A: Numeric, B: Numeric, C: Numeric> Payload<OperandTriple<A, B, C>>
    for OperandTriple<A, B, C>
{
    fn operands(&self) -> comptime_type!(Vec<StageOperand>) {
        let a = self.a.delivery();
        let b = self.b.delivery();
        let c = self.c.delivery();
        comptime!(vec![
            StageOperand {
                delivery: a,
                space: self.a.place.space.clone(),
            },
            StageOperand {
                delivery: b,
                space: self.b.place.space.clone(),
            },
            StageOperand {
                delivery: c,
                space: self.c.place.space.clone(),
            },
        ])
    }

    fn staged(
        &self,
        first: &OperandTriple<A, B, C>,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> OperandTriple<A, B, C> {
        let a = if comptime!(refills[0] == Refill::Shared) {
            first.a.clone()
        } else {
            stage_one(&self.a, comptime!(spec.clone()))
        };
        let b = if comptime!(refills[1] == Refill::Shared) {
            first.b.clone()
        } else {
            stage_one(&self.b, comptime!(spec.clone()))
        };
        let c = if comptime!(refills[2] == Refill::Shared) {
            first.c.clone()
        } else {
            stage_one(&self.c, comptime!(spec))
        };
        OperandTriple::<A, B, C> { a, b, c }
    }

    fn bring(
        &mut self,
        src: &OperandTriple<A, B, C>,
        meeting: &Meeting,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
        #[comptime] only: Refill,
    ) {
        if comptime!(refills[0] == only) {
            meeting.fill(&mut self.a, &src.a.at(region));
        }
        if comptime!(refills[1] == only) {
            meeting.fill(&mut self.b, &src.b.at(region));
        }
        if comptime!(refills[2] == only) {
            meeting.fill(&mut self.c, &src.c.at(region));
        }
    }
}
