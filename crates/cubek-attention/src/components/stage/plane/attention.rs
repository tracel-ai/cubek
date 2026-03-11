use cubecl;
use cubecl::prelude::*;

use crate::components::{
    global::simple::PlaneAttentionWriter,
    softmax::BroadcastReducer,
    stage::{partition_attention::PartitionAttention, partitioner::AttentionPartitioner},
    tile::TileAttentionConfig,
};

use crate::components::stage::SharedPartitionAttentionConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlanePartitionStageConfig<TC: TileAttentionConfig> {
    pub shared: SharedPartitionAttentionConfig<TC>,
}

pub type PlanePartitionAttention<AP, SK, SV, SO, TA> =
    PartitionAttention<AP, SK, SV, SO, TA, PlanePartitioner>;

pub struct PlanePartitioner {}

#[cube]
impl AttentionPartitioner for PlanePartitioner {
    type Reducer = BroadcastReducer;
    type Writer<ES: Float, EG: Float> = PlaneAttentionWriter<ES, EG>;

    fn seq_q_index() -> u32 {
        UNIT_POS_Y
    }
}
