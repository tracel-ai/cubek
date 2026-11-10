//! The interpolate routines on the tile stack: each places the cube interpolate microkernel
//! (`cubek_microkernel::interpolate`) over an NHWC resize and launches it. The problem, its
//! tensors and their binding, the filters a kernel weighs the taps with, the selector, and the
//! routines, one folder each.
//!
//! Each output position expands into a square of source taps. A gathered input tile holds those
//! samples, a procedural tile supplies separable filter weights, and the microkernel reduces the
//! taps into the output's channels.

mod coordinate;
mod filter;
mod geometry;
mod launch;
mod problem;
mod routines;
mod select;
mod space;
mod strategy;

pub use launch::*;
pub use problem::*;
pub use routines::*;
pub use select::select;
pub use strategy::*;
