//! The two-operand payload: what a contraction's slots hold.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

/// Two operands staged together; `Clone` duplicates handles, not buffers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct OperandPair<Lhs: Numeric, Rhs: Numeric> {
    pub lhs: Tile<Lhs>,
    pub rhs: Tile<Rhs>,
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Payload<OperandPair<Lhs, Rhs>> for OperandPair<Lhs, Rhs> {
    fn operands(&self) -> comptime_type!(Vec<StageOperand>) {
        let lhs = self.lhs.delivery();
        let rhs = self.rhs.delivery();
        // The scales a stage keeps beside its values come after both operands, so each operand
        // keeps its place in the slot's refills.
        let lhs_scales = self.lhs.scale_operands();
        let rhs_scales = self.rhs.scale_operands();
        comptime!(
            [
                StageOperand {
                    delivery: lhs,
                    space: self.lhs.place.space.clone(),
                },
                StageOperand {
                    delivery: rhs,
                    space: self.rhs.place.space.clone(),
                },
            ]
            .into_iter()
            .chain(lhs_scales)
            .chain(rhs_scales)
            .collect()
        )
    }

    fn staged(
        &self,
        first: &OperandPair<Lhs, Rhs>,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> OperandPair<Lhs, Rhs> {
        let lhs = if comptime!(refills[0] == Refill::Shared) {
            first.lhs.clone()
        } else {
            stage_one(&self.lhs, comptime!(spec.clone()))
        };
        let rhs = if comptime!(refills[1] == Refill::Shared) {
            first.rhs.clone()
        } else {
            stage_one(&self.rhs, comptime!(spec))
        };
        OperandPair::<Lhs, Rhs> { lhs, rhs }
    }

    fn free(&self) {
        self.lhs.free_stage();
        self.rhs.free_stage();
    }

    fn bring(
        &mut self,
        src: &OperandPair<Lhs, Rhs>,
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
