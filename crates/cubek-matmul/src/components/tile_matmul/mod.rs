//! Matrix multiplication on register- or shared-memory tiles.
//! Optimized for fixed shapes and low-level compute strategies.

mod base;
mod config;
mod dispatch;
mod tile;

pub use base::*;
pub use config::*;
pub use dispatch::*;
pub use tile::*;
