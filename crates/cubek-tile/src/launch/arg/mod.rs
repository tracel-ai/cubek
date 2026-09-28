//! The kernel argument that carries one operand, host and device side: what is bound
//! ([`Input`], [`Output`]) and the arguments they carry, one file each.

mod accumulate;
mod analysis;
mod base;
mod delivery;
mod operand;
mod scale;
mod tensor;
mod tma;

pub use accumulate::*;
pub use analysis::Refusal;
pub use base::*;
pub use delivery::*;
pub use operand::*;
pub use scale::*;
pub use tensor::*;
pub use tma::*;
