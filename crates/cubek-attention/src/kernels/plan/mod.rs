//! How an attention runs, as a plan ([`cubek_std::plan`]): the kernels a launch tries, in order,
//! and the cache rebuilt as floats before them, on one [`AttentionCall`]. A measured choice is the
//! caller's plan, handed to [`select`](super::select::selector::select).

mod call;
mod kernel;

pub use call::*;
pub use kernel::*;
