/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

pub mod gemm;
pub mod gemv_unit_perpendicular;

/// The cooperative, tiled `BatchMatmulRoutine` family sharing one launch hub.
pub mod batch;

mod selector;

pub use selector::*;
