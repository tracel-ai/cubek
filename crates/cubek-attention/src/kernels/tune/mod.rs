//! How the attention family is measured: the tuning [`key`] that makes two attentions the same
//! attention, and the candidate every routine is listed as. A caller builds the tables from
//! them.

mod candidate;
pub mod key;

pub use candidate::*;
