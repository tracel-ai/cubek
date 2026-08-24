//! The matmul problem, its element types, and the errors either family can raise.

mod base;
mod cost;
mod elems;
mod error;
mod vectorization;

pub use base::*;
pub use cost::*;
pub use elems::*;
pub use error::*;
pub use vectorization::*;

// Internal-only — external crates import this directly from cubek-std.
pub(crate) use cubek_std::StageIdent;
