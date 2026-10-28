//! How a product runs, as a plan ([`cubek_std::plan`]): the routines that read it, the decode
//! that comes before them, in the order a launch tries them, on one [`MatmulCall`].

mod call;
mod decoded;
mod routine;

pub use call::*;
pub use decoded::*;
pub use routine::*;
