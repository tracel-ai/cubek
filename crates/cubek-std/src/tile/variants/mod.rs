//! Per-variant tile data, configs, and per-variant compute (matmul-execute,
//! load/write/zero-init, readers/writers, copy_from). Cross-variant dispatch
//! lives in [`crate::tile::ops`]; cross-variant support types live as
//! siblings ([`crate::tile::row_wise`], [`crate::tile::scope`]).
//!
//! Layout:
//! - [`instruction`] — variants tied to a tile-level matmul instruction
//!   (cmma, mma, register, plane_vec, interleaved).
//! - [`stage`] — variants wrapping stage memory and per-partition tile
//!   collections (added in PR 2).
//! - Flat root — non-instruction, non-stage variants (`bounce`, `unit`,
//!   `whitebox_fragment`, `strided`, `kind`, `row_wise`).

pub(crate) mod bounce;
pub(crate) mod instruction;
pub(crate) mod kind;
pub(crate) mod row_wise;
pub(crate) mod stage;
pub(crate) mod strided;
pub(crate) mod unit;
pub(crate) mod whitebox_fragment;

pub use bounce::*;
pub use instruction::*;
pub use kind::*;
pub use row_wise::*;
pub use stage::*;
pub use strided::*;
pub use unit::*;
pub use whitebox_fragment::*;
