//! The axis-agnostic tile DSL engine
#![allow(dead_code)]

mod layout;
mod load;
mod matmul;
mod partitioner;
mod quantization;
mod resident;
mod scalar;
mod space;
mod stage;
mod tile;
mod vec_tensor;
mod view;

// `Axis`/`MAX_AXES` and `ConcreteLayout` are the storage-layout vocabulary; clients reach them
// through `cubek_tile::{Axis, ...}`.
pub use layout::*;
pub use load::*;
pub use partitioner::*;
pub use quantization::*;
pub use scalar::*;
pub use space::*;
pub use stage::*;
pub use tile::*;
pub use vec_tensor::*;
pub use view::*;
