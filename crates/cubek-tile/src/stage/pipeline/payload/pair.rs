//! The two-operand payload: what a contraction's slots hold.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

/// Two operands staged together: what a contraction's slots hold.
///
/// A named pair rather than a tuple, because a `#[cube]` trait is implemented for a named type.
/// `Clone` duplicates the handles, not the buffers.
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
