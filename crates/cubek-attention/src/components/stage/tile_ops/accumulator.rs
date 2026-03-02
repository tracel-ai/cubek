use cubecl;
use cubecl::prelude::*;

use crate::components::tile::RowWise;
use crate::components::tile::TileAttention;
use crate::components::tile::{AccumulatorPipeline, AccumulatorPipelineExpand};
use crate::components::tile::{AccumulatorRowwise, AccumulatorRowwiseExpand};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::SM;

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct AccumulatorTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Accumulator,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> AccumulatorTile<AP, TA> {
    pub fn new(
        shared: &mut TA::AccumulatorShared,
        #[comptime] config: TA::Config,
    ) -> AccumulatorTile<AP, TA> {
        let mut fragment = TA::allocate_accumulator(shared, config);
        fragment.zero();

        AccumulatorTile::<AP, TA> { fragment }
    }
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> AccumulatorTile<AP, TA> {
    /// Multiplies each row by a scale
    pub fn scale_mul(&mut self, scale: &RowWise<SM<AP>>) {
        self.fragment
            .rowwise_mut()
            .rowwise_scale(&RowWise::<SM<AP>>::cast_from(scale));
        self.fragment.finalize_acc();
    }

    /// Divides each row by a scale
    pub fn scale_div(&mut self, scale: &RowWise<SM<AP>>) {
        let mut scale = RowWise::<SM<AP>>::cast_from(scale);
        scale.recip_inplace();
        self.fragment
            .rowwise_mut()
            .rowwise_scale(&RowWise::<SM<AP>>::cast_from(&scale));
        self.fragment.finalize_acc();
    }
}
