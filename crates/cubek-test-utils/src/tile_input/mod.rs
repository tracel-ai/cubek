//! A clean, launchable [`Tile`](cubecl::tile::Tile) input for tests. Describe a tile as a
//! [`Space`](cubecl::tile::Space)
//! with a stack of `.split`/`.tile` levels and get a launchable tile: the buffer
//! is a plain `[grid…, tile…]` strided tensor (or `[grid…, level1…, …]` when
//! recursively tiled), presented in its logical space via an explicit
//! `tiled_view`. No `Tiler`, no semantic-view juggling.
#![allow(dead_code)]

mod base;
mod builder;
mod quantized;
mod quantized_builder;

pub use base::*;
pub use builder::*;
pub use quantized::*;
pub use quantized_builder::*;
