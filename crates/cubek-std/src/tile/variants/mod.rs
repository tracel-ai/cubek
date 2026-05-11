//! Per-variant tile data, configs, and per-variant compute (matmul-execute,
//! load/write/zero-init, readers/writers, copy_from). Cross-variant dispatch
//! lives in [`crate::tile::ops`]; cross-variant support types live as
//! siblings ([`crate::tile::row_wise`], [`crate::tile::scope`]).

pub(crate) mod bounce;
pub(crate) mod cmma;
pub(crate) mod interleaved;
pub(crate) mod kind;
pub(crate) mod mma;
pub(crate) mod plane_vec;
pub(crate) mod register;
pub(crate) mod row_wise;
pub(crate) mod strided;
pub(crate) mod unit;
pub(crate) mod whitebox_fragment;

pub use bounce::*;
pub use cmma::*;
pub use interleaved::*;
pub use kind::*;
pub use mma::*;
pub use plane_vec::*;
pub use register::*;
pub use row_wise::*;
pub use strided::*;
pub use unit::*;
pub use whitebox_fragment::*;
