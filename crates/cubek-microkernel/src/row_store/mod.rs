//! Where a producer of rows writes them: as computed, or packed under block scales along each row,
//! in the form the lhs of a product reads ([`RowStore`]).
//!
//! The RMS norm and the SwiGLU gate are the producers. Both write a row a line at a time from the
//! registers of the units that own it ([`OwnerLines`]), and a cube's owners write through one
//! [`CubeStore`]: straight into the output, or into a stage of the cube's box that the cube encodes
//! into the packed output once every owner has written ([`Encoded`]). So a producer has no
//! packed twin: the same walk writes either, and what the output is decides which.
//!
//! A plain store writes its rows wherever its output's binding lays them: in rows, or in the
//! storage tiles a product reads its operand from, the tiling riding the binding, so nothing is
//! relaid out after.

mod base;
mod cube_store;

pub use base::*;
pub use cube_store::*;
