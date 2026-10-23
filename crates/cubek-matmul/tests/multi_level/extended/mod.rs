//! Extended tier: hand-written forced-blueprint tests covering harder or
//! niche cases: per-routine TilingScheme sweep, alt shapes, non-default
//! layouts, hypercube / swizzle / specialization / partition-buffering knobs,
//! and quantization.

mod common;

mod advanced;
mod alt_shapes;
mod gemm;
mod layouts;
mod quantization;
mod stride_zero;
mod tiling_scheme;
