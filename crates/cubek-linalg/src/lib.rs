//! This provides different implementations linear algebra algorithms
//!
//! The crate follows the blueprint-routine architecture:
//! - [`definition`] holds the runtime problem descriptions and errors.
//! - [`routines`] adapts each algorithm to the hardware, producing a minimal
//!   comptime `Blueprint` plus runtime launch settings.
//! - [`components`] holds the kernels, specialized by their blueprint.
//! - [`launch`](mod@launch) validates the input and dispatches a
//!   [`QRStrategy`] to the matching routine and component.

pub mod components;
pub mod definition;
pub mod launch;
pub mod routines;

pub use definition::*;
pub use launch::*;
