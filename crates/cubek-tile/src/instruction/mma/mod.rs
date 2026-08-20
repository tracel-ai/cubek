//! Hardware CMMA/WMMA, manual MMA, and software outer-product GEMM microkernels.

mod base;
mod cmma;
mod manual;
pub(crate) mod packed;
pub(crate) mod register;

pub use base::*;
