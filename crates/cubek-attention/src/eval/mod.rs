//! CPU references and benchmark catalogues for attention.
//!
//! Layout:
//! - [`benchmarks`] — pieces shared by forward and backward catalogues
//!   (problem definitions, batches × heads × seq grid).
//! - [`forward`] — CPU reference + benchmark catalogue for the forward pass.
//! - [`backward`] — CPU reference + benchmark catalogue for the backward pass.

#[cfg(feature = "benchmarks")]
pub mod problem;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod backward;
#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod forward;
