/// Tensor argument DSL (`MatmulArgs` and friends) shared by the launch paths.
pub mod args;
/// Components for matrix multiplication
pub mod components;
pub mod definition;
/// The public launch entry point.
pub mod launch;
/// Contains matmul kernels
pub mod routines;
/// Strategy selection, autotune keys, and kernel dispatch.
pub mod strategy;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;
