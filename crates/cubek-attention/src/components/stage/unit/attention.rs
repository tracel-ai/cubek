use cubecl;
use cubecl::prelude::*;

use crate::components::{
    global::simple::UnitAttentionWriter,
    stage::{partition_attention::PartitionAttention, partitioner::AttentionPartitioner},
    tile::TileAttentionConfig,
};

use crate::components::stage::SharedPartitionAttentionConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitPartitionStageConfig<TC: TileAttentionConfig> {
    pub shared: SharedPartitionAttentionConfig<TC>,
}

pub type UnitPartitionAttention<AP, SK, SV, SO, TA> =
    PartitionAttention<AP, SK, SV, SO, TA, UnitPartitioner>;

pub struct UnitPartitioner {}

#[cube]
impl AttentionPartitioner for UnitPartitioner {
    type Writer<ES: Float, ESS: Size, EG: Float, EGS: Size> = UnitAttentionWriter<ES, ESS, EG, EGS>;

    fn seq_q_index() -> u32 {
        UNIT_POS
    }
}
