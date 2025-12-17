//! Executes multiple independent global matmuls with optional broadcasting.

pub mod naive;

mod base;
mod layout;
mod partitioned_matmul;

pub use base::*;
pub use layout::*;
pub use partitioned_matmul::*;
