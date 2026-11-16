use crate::attention::{DV, S};
use cubecl::ir::ElemType;
use cubecl::tile::Tile;
use cubecl::tile::ops::softmax::OnlineSoftmax;
use cubecl::{self as cubecl, cube, prelude::*};
use serde::{Deserialize, Serialize};

/// How a query's scores become the weights its values are summed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionWeighting {
    /// `softmax(score)` along the keys: each weight against the rest of its row, the rows summing
    /// to one.
    Softmax,
    /// `max(score, 0)`: each weight on its own, nothing normalized, as a key selector scores
    /// cache entries against its heads.
    Relu,
}

impl AttentionWeighting {
    /// 32-bit registers one unit keeps this weighting's state of `rows` rows in, summed in `sum`:
    /// the softmax's running max and sum of each; nothing for a relu, which weighs each score on
    /// its own.
    pub fn state_registers(self, rows: usize, sum: ElemType) -> u32 {
        match self {
            Self::Softmax => (2 * rows * sum.size()).div_ceil(4) as u32,
            Self::Relu => 0,
        }
    }
}

/// One block of `score` taken in as `weighting` weighs it, where its holder keeps it: the softmax
/// moving its rows on ([`OnlineSoftmax::step`]), or each score's relu
/// ([`Tile::relu_weights`]). Returns what the rows' sum against the earlier blocks is multiplied
/// by, which only a softmax has; `score` is left holding the block's weights, read from the tile
/// [`Tile::scaled`] was called on where `score` carries a factor.
#[cube]
pub fn weigh<EA: Float>(
    score: &Tile,
    softmax: &mut OnlineSoftmax<EA>,
    bias: &Tile,
    scale: EA,
    #[comptime] weighting: AttentionWeighting,
) -> ComptimeOption<Array<EA>> {
    match weighting {
        AttentionWeighting::Softmax => ComptimeOption::new_Some(softmax.step(score, bias, scale)),
        AttentionWeighting::Relu => {
            score.relu_weights::<EA>(bias, scale, S);
            ComptimeOption::new_None()
        }
    }
}

/// `sum`'s rows multiplied by `factors` along `DV`, where there are any: a softmax's correction or
/// normalizer, which a relu weighting has none of.
#[cube]
pub fn rescale<EA: Float>(sum: &mut Tile, factors: ComptimeOption<Array<EA>>) {
    #[comptime]
    match factors {
        ComptimeOption::Some(factors) => sum.along(DV).mul(&factors),
        ComptimeOption::None => {}
    }
}
