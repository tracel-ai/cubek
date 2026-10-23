//! How the attention family is measured: the tuning [`key`] that makes two attentions the same
//! attention, the candidate every routine is listed as, and the autotunes that race them
//! ([`launch_tuned`]).

mod base;
mod candidate;
pub mod key;

pub use base::*;
pub use candidate::*;
