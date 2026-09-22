//! The [`Walk`]: the regions one level of a space hands the instance running the code.

mod base;
mod deal;
mod order;
mod portion;

pub use base::*;
pub(crate) use deal::AxisDeal;
pub use order::*;
pub use portion::*;
