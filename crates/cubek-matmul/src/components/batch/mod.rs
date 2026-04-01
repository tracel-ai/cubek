//! Executes multiple independent global matmuls with optional broadcasting.

pub mod naive;
pub mod vecmat_plane_parallel;
pub mod vecmat_unit_perpendicular;

mod base;
mod layout;
mod partitioned_matmul;

pub use base::*;
pub use layout::*;
pub use partitioned_matmul::*;
