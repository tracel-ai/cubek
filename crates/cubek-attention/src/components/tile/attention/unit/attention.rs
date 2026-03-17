use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::{
        TileAttention, attention::unit::setup::UnitTileAttentionConfig, matmul::UnitMatmul,
        output::unit::UnitAttentionOutput, softmax::unit::UnitSoftmax,
    },
    definition::{AttentionPrecision, attention_types::*},
};

pub struct UnitTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for UnitTileAttention {
    type Config = UnitTileAttentionConfig;
    type ScoreMatmul = UnitMatmul<QT<AP>, KVT<AP>, SM<AP>>;
    type Softmax = UnitSoftmax<SML<AP>>;
    type ValueMatmul = UnitMatmul<SML<AP>, KVT<AP>, ACC<AP>>;
    type Output = UnitAttentionOutput<SM<AP>, ACC<AP>>;
}
