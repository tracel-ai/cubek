//! Linear algebra kernels for cubek.
//!
//! QR decomposition is implemented with the TSQR-inspired blocked Householder
//! routine ([`routines::BahtTsqrRoutine`]), the fastest of the strategies
//! benchmarked for this crate. The crate follows the blueprint-routine
//! architecture:
//! - [`definition`] holds the runtime problem descriptions and errors.
//! - [`routines`] adapts the algorithm to the hardware, producing a minimal
//!   comptime `Blueprint` plus runtime launch settings.
//! - [`components`] holds the kernels, specialized by their blueprint.
//! - [`launch`](mod@launch) validates the input and dispatches to the
//!   matching routine and component.

pub mod components;
pub mod definition;
pub mod eval;
pub mod launch;
pub mod routines;

pub use definition::*;
pub use launch::*;
