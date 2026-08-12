//! Hardware CMMA/WMMA, manual MMA, and software outer-product GEMM microkernels.

mod base;
mod cmma;
mod manual;
mod register;

pub use base::*;
