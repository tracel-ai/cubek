/// Accelerated but using shared memory for rowwise operations
pub mod blackbox_accelerated;
/// Unit attention
pub mod unit;
/// Accelerated and doing arithmetic directly on fragments
pub mod whitebox_accelerated;

mod base;

pub use base::*;
