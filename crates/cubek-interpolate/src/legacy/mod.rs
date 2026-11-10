//! The backward kernels, still written on cubecl directly rather than on routines.

mod bilinear_backward;
mod nearest_backward;

pub(crate) use bilinear_backward::interpolate_bilinear_backward_launch;
pub(crate) use nearest_backward::interpolate_nearest_backward_launch;
