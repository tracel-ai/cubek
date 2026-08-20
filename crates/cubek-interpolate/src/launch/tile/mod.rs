//! Tile-backed interpolation as a separable gather-reduce.
//!
//! Each output position expands into a square of source taps. A gathered input tile holds those
//! samples, a procedural tile supplies separable filter weights, and the ordinary tile MMA
//! reduces the taps into NHWC output channels.

mod base;
pub(crate) mod coordinate;
mod filter;
mod geometry;
mod kernel;
pub(crate) mod space;

pub use base::*;
pub use geometry::*;
