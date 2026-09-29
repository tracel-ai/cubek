//! Shared arithmetic: [`Monoid`], [`Semiring`] and constant-keeping `Integer`.

pub(crate) mod constant;
pub(crate) mod monoid;
pub(crate) mod semiring;

pub(crate) use constant::Integer;
pub(crate) use constant::{
    IntegerExpand, IntegerSeq, IntegerSeqExpand, constant, fold_add, fold_mul,
};
pub(crate) use monoid::comptime_only;
