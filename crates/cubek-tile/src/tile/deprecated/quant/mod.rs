//! The scales-at-the-read path: a dequantizing view inside the memory read ([`info`], [`view`])
//! and the launch argument that carries values, scales and scheme as one ([`arg`]).
//!
//! Deprecated: a scale is a tile, and a scaled operand is a multiply (`Tile::mul`). This folder
//! exists until lora_linear and attention_mb's packed fold read two tiles and write the multiply;
//! nothing new should reach for it.

mod arg;
mod info;
mod view;

pub use arg::*;
pub use info::*;
pub use view::*;
