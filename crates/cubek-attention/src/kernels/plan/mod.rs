//! How an attention runs, as a plan ([`cubek_std::plan`]): the kernels a launch tries, in order,
//! and the cache rebuilt as floats before them, on one [`AttentionCall`]. The autotune it measures
//! on is [`tune`](super::tune)'s ([`fastest`](super::tune::fastest)).

mod call;
mod kernel;

pub use call::*;
pub use kernel::*;
