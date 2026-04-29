mod broadcast_reducer;
mod local_tile;
mod mask;
mod mask_fragment;
mod rowwise;
mod softmax;
mod tile_ops;
mod unit_tile;
mod workspace;

pub use broadcast_reducer::*;
pub use local_tile::*;
pub use mask::*;
pub use mask_fragment::*;
pub use rowwise::*;
pub use softmax::*;
pub use unit_tile::*;
pub use workspace::*;

/// Logits below this are considered masked (effectively -inf)
/// Value chosen to fit within f16 range (~-65,504 max)
pub const LOGIT_MASKED: f32 = -6e4;

/// Any value smaller than this is considered numerically zero
/// (used for fully-masked rows or tiny contributions)
/// Value chosen to be above f16 smallest normal (~6.1e-5)
pub const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;
