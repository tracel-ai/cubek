//! Per-flavor tile matmul compute: `*_execute`, `*_load_*`, `*_write_to_shared`,
//! plus the fragment readers/writers for each flavor. Tile data and matmul
//! configs live alongside the corresponding data structures in
//! [`crate::tile::data`].

pub mod cmma;
pub mod interleaved;
pub mod mma;
pub mod plane_vec;
pub mod register;

pub use cmma::*;
pub use interleaved::*;
pub use mma::*;
pub use plane_vec::*;
pub use register::*;
