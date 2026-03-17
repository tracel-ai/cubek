/// Accelerated but using shared memory for rowwise operations
pub mod blackbox_accelerated;
/// Unit attention
pub mod unit;

mod base;

pub use base::*;
