use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::{
        TileAttention, attention::blackbox::setup::BlackboxAcceleratedAttentionConfig,
        matmul::CmmaMatmul, output::blackbox::BlackboxAttentionOutput,
        softmax::blackbox::BlackboxSoftmax,
    },
    definition::{AttentionPrecision, attention_types::*},
};

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is blackbox
pub struct BlackboxAcceleratedTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for BlackboxAcceleratedTileAttention {
    type Config = BlackboxAcceleratedAttentionConfig;
    type ScoreMatmul = CmmaMatmul;
    type Softmax = BlackboxSoftmax<SML<AP>>;
    type ValueMatmul = CmmaMatmul;
    type Output = BlackboxAttentionOutput<SM<AP>, ACC<AP>>;
}
