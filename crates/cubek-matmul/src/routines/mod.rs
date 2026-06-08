/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

pub mod gemm;
pub mod gemv_unit_perpendicular;
pub mod mosaic;

pub mod double_buffering;
pub mod double_unit;
pub mod gemv_innerproduct;
pub mod interleaved;
pub mod ordered_double_buffering;
pub mod simple;
pub mod simple_unit;
pub mod specialized;

mod base;
mod selector;

pub use base::*;
pub use selector::*;
