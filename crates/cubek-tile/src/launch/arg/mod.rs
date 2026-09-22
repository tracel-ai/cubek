//! The kernel argument that carries one operand, host and device side, one file per delivery.

mod accumulate;
mod base;
mod delivery;
mod labels;
mod quant;
mod tensor;
mod tma;

pub use accumulate::*;
pub use base::*;
pub use delivery::*;
pub use labels::*;
pub use quant::*;
pub use tensor::*;
pub use tma::*;
