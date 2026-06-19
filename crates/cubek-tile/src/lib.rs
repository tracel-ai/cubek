//! The axis-agnostic tile DSL engine
#![allow(dead_code)]

mod matmul;
mod partitioner;
mod payload;
mod quantization;
mod ring;
mod space;
mod tile;
mod view;

// The layout-request vocabulary and `Axis`/`MAX_AXES` live in the leaf `cubek-layout` crate,
// re-exported here so tile-engine code and clients keep using `cubek_tile::{Axis, ...}`.
pub use cubek_layout::*;
pub use partitioner::*;
pub use payload::*;
pub use quantization::*;
pub use ring::*;
pub use space::*;
pub use tile::*;
pub use view::*;
