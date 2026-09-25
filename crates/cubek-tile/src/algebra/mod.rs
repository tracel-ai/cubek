//! The arithmetic every leaf shares: a [`Monoid`] to fold values together, a [`Semiring`] pairing
//! one with the product a contraction forms first, and [`Integer`], integer arithmetic on kernel
//! values that keeps a constant constant.

mod constant;
mod monoid;
mod semiring;

pub use constant::Integer;
pub(crate) use constant::{
    IntegerExpand, IntegerSeq, IntegerSeqExpand, constant, fold_add, fold_mul,
};
pub(crate) use monoid::comptime_only;
pub use monoid::{Carrier, Monoid};
pub use semiring::Semiring;
