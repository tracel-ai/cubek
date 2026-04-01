/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

pub mod vecmat_plane_parallel;
pub mod vecmat_unit_perpendicular;

pub mod double_buffering;
pub mod double_unit;
pub mod interleaved;
pub mod ordered_double_buffering;
pub mod simple;
pub mod simple_unit;
pub mod specialized;
pub mod vecmat_innerproduct;

mod base;
mod selector;

pub use base::*;
pub use selector::*;
