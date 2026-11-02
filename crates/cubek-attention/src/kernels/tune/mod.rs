//! How the attention family is measured: the tuning [`key`] that makes two attentions the same
//! attention, the candidate every routine is listed as, and the autotunes that race them
//! ([`launch_tuned`]).

mod autotunes;
mod candidate;
mod inputs;
pub mod key;

pub use autotunes::*;
pub use candidate::*;
pub use inputs::*;
