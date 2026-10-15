//! The one-operand payload.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

#[cube]
impl Payload<Tile> for Tile {
    fn operands(&self) -> comptime_type!(Vec<StageOperand>) {
        let delivery = self.delivery();
        let scales = self.scale_operands();
        comptime!(
            [StageOperand {
                delivery,
                space: self.place.space.clone(),
            }]
            .into_iter()
            .chain(scales)
            .collect()
        )
    }

    fn staged(
        &self,
        first: &Tile,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> Tile {
        if comptime!(refills[0] == Refill::Shared) {
            first.clone()
        } else {
            stage_one(self, comptime!(spec))
        }
    }

    fn free(&self) {
        self.free_stage();
    }

    fn bring(
        &mut self,
        src: &Tile,
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
