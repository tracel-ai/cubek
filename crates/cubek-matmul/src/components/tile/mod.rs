//! Matrix multiplication on register- or shared-memory tiles.
//! Optimized for fixed shapes and low-level compute strategies.

pub mod cmma;
pub mod interleaved;
pub mod mma;
pub mod plane_vec_mat_inner_product;
pub mod register;

mod base;
mod config;
mod tile;

pub use base::*;
pub use config::*;
pub use tile::*;
