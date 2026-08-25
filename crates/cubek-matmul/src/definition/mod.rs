//! The matmul problem, its element types, and the errors either architecture can raise.

mod base;
mod cost;
mod elems;
mod error;
mod precision;
mod vectorization;

pub use base::*;
pub use cost::*;
pub use elems::*;
pub use error::*;
pub use precision::*;
pub use vectorization::*;

// Internal-only: external crates import this directly from cubek-std.
#[cfg(feature = "multi-level")]
pub(crate) use cubek_std::StageIdent;
