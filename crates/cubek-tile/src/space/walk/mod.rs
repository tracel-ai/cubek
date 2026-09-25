//! The [`Walk`]: the regions one level of a space hands the instance running the code.

mod base;
mod distribution;
mod order;
mod portion;

pub use base::*;
pub(crate) use distribution::AxisDistribution;
pub use order::*;
pub use portion::*;
