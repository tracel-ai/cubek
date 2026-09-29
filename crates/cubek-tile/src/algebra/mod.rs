//! Shared arithmetic: [`Monoid`], [`Semiring`] and constant-keeping [`Integer`].

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
