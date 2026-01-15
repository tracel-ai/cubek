use cubecl;
use cubecl::prelude::*;

use crate::components::{
    global::simple::UnitAttentionWriter,
    stage::{
        UnitReducer, partition_attention::PartitionAttention, partitioner::AttentionPartitioner,
    },
};

use crate::components::{stage::SharedPartitionAttentionConfig, tile::TileAttentionConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitPartitionStageConfig<TC: TileAttentionConfig> {
    pub shared: SharedPartitionAttentionConfig<TC>,
}

pub type UnitPartitionAttention<AP, SK, SV, SO, TA> =
    PartitionAttention<AP, SK, SV, SO, TA, UnitPartitioner>;

pub struct UnitPartitioner {}

#[cube]
impl AttentionPartitioner for UnitPartitioner {
    type Reducer = UnitReducer;
    type Writer<ES: Float, EG: Float> = UnitAttentionWriter<ES, EG>;

    fn seq_q_index() -> u32 {
        UNIT_POS
    }
}
