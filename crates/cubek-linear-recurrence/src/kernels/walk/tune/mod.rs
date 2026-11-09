//! How the walk is measured: the tuning [`key`] that makes two walks the same walk, the candidate
//! the cube routine is listed as, and the autotune that races them ([`launch_tuned`]).
//!
//! The table is not bounded: the walk reads the queries, keys and values the step just wrote, so
//! production meets them warm, and a round read cold would time a kernel nobody runs.

mod autotunes;
mod candidate;
pub mod key;

pub use autotunes::*;
pub use candidate::*;
