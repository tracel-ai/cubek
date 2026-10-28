//! How an attention runs, as a plan ([`cubek_std::plan`]): the kernels a launch tries, in order,
//! the autotune it measures on, and the cache rebuilt as floats before them, on one
//! [`AttentionCall`].

mod call;
mod kernel;

pub use call::*;
pub use kernel::*;
