use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::{
        TileAttention, attention::unit::setup::UnitTileAttentionConfig,
        output::unit::UnitAttentionOutput, softmax::unit::UnitSoftmax,
    },
    definition::{AttentionPrecision, attention_types::*},
};

pub struct UnitTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for UnitTileAttention {
    type Config = UnitTileAttentionConfig;
    type Softmax = UnitSoftmax<SML<AP>>;
    type Output = UnitAttentionOutput<SM<AP>, ACC<AP>>;
}
