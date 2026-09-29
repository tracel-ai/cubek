//! The kernel argument that carries one operand, host and device side: what is bound
//! ([`Input`], [`Output`]), the arguments they carry, one file each, and the [`Destination`] seam
//! a caller's own output plugs into.

mod accumulate;
mod analysis;
mod base;
mod delivery;
mod destination;
mod operand;
mod scale;
mod tensor;
mod tma;

pub use accumulate::*;
pub use analysis::Refusal;
pub use base::*;
pub use delivery::*;
pub use destination::*;
pub use operand::*;
pub use scale::*;
pub use tensor::*;
pub use tma::*;
