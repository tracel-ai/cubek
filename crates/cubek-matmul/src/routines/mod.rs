/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

pub mod cpu_gemm;
pub mod cyclic_cmma;
pub mod gemm;
pub mod gemv_unit_perpendicular;

/// The cooperative, tiled `BatchMatmulRoutine` family sharing one launch hub.
pub mod batch;

/// TMA descriptor geometry for tile-DSL operands.
pub mod tma_tile;

mod base;
mod selector;

pub use base::*;
pub use selector::*;
