//! A 2D convolution as one tile contraction over a gathered input. Forward is `out[b, oh, ow, co]
//! += input[b, oh*s + rh*d - p, ow*s + rw*d - p, ci] * weight[co, rh, rw, ci]`, contracted over
//! the taps and the channels; the weight gradient is `weight_grad[co, rh, rw, ci] += input[..] *
//! out_grad[b, oh, ow, co]`, contracted over the batch and the output positions; the data
//! gradient reads the output gradient backward through the taps, at stride 1. All are one
//! space and one projection of the input, with the roles of the axes moved ([`ConvOperation`]).
//!
//! The routines in `routines` are what runs, one folder per scope. Activations are NHWC and the
//! weight and its gradient `[co, kh, kw, ci]`.

mod binding;
mod blueprint;
mod launch;
mod operation;
mod partitioning;
mod problem;
mod routines;
mod tensors;
mod window;

pub use binding::*;
pub use blueprint::*;
pub use launch::*;
pub use operation::*;
pub use partitioning::*;
pub use problem::*;
pub use routines::*;
pub use tensors::*;
pub use window::*;
