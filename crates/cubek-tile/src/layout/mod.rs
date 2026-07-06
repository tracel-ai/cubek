//! The physical (storage) layout vocabulary: the labeled [`Axis`] a buffer ranges over and the
//! [`ConcreteLayout`] a stored buffer realizes. Formerly the standalone `cubek-layout` crate,
//! folded in since the tile engine is its only consumer.

mod axis;
mod concrete;

pub use axis::*;
pub use concrete::*;
