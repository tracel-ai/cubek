/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

pub mod cpu_gemm;
pub mod gemm;
/// Builds a matmul operand's TMA descriptor, then loads it via [`TileArgLaunch::tma`](cubek_tile::TileArgLaunch::tma).
pub mod tma_tile;
pub mod gemv_unit_perpendicular;

/// The cooperative, tiled `BatchMatmulRoutine` family sharing one launch hub.
pub mod batch;

mod base;
mod selector;

pub use base::*;
pub use selector::*;
