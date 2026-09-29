//! The [`Walk`]: the regions one level of a space hands the instance running the code.

pub(crate) mod base;
pub(crate) mod distribution;
pub(crate) mod order;
pub(crate) mod portion;

pub use base::*;
pub(crate) use distribution::AxisDistribution;
pub(crate) use order::*;
