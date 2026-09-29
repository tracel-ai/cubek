//! The one-operand payload.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

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
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> Tile<T> {
        if comptime!(refills[0] == Refill::Shared) {
            first.clone()
        } else {
            stage_one(self, comptime!(spec))
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
