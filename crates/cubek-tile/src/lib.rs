//! The axis-agnostic tile DSL engine
#![allow(dead_code)]

mod fold;
mod ops;
mod physical;
mod space;
mod staging;
mod tile;

// `Axis`/`MAX_AXES` and `ConcreteLayout` are the storage-layout vocabulary; clients reach them
// through `cubek_tile::{Axis, ...}`.
pub use fold::*;
pub use ops::*;
pub use physical::*;
pub use space::*;
pub use staging::*;
pub use tile::*;
