//! A 2D convolution as one tile contraction over a gathered input: the forward, the weight
//! gradient and the data gradient share one space and one projection of the input, with the roles
//! of the axes moved ([`ConvolutionOperation`]). The data gradient is at stride 1 only.
//!
//! Activations are NHWC and the weight and its gradient `[co, kh, kw, ci]`. The routines in
//! `routines` are what runs, one folder per scope.

mod binding;
mod blueprint;
mod declined;
mod launch;
mod partitioning;
mod problem;
mod roles;
mod routines;
mod staging;
mod tensors;
mod tune;

pub use binding::*;
pub use blueprint::*;
pub use declined::*;
pub use launch::*;
pub use partitioning::*;
pub use problem::*;
pub use roles::*;
pub use routines::*;
pub use staging::*;
pub use tensors::*;
pub use tune::*;
