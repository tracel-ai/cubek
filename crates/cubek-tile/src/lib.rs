//! The axis-agnostic tile DSL engine
#![allow(dead_code)]

mod layout;
mod load;
mod matmul;
mod partitioner;
mod quantization;
mod ring;
mod space;
mod tile;
mod tile_kind;
mod view;

// `Axis`/`MAX_AXES` and `ConcreteLayout` are the storage-layout vocabulary; clients reach them
// through `cubek_tile::{Axis, ...}`.
pub use layout::*;
pub use load::*;
pub use partitioner::*;
pub use quantization::*;
pub use ring::*;
pub use space::*;
pub use tile::*;
pub use tile_kind::*;
pub use view::*;
