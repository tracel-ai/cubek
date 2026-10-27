//! CPU references and benchmark catalogues for attention.
//!
//! Layout:
//! - [`problem`]: problem definitions, the batches × heads × seq grid.
//! - [`forward`]: CPU reference + benchmark catalogue for the forward pass.

#[cfg(feature = "benchmarks")]
pub mod problem;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod forward;
