//! Matmul routines written on the tile DSL.

pub mod cmma;
pub mod cpu_gemm;
pub mod quant_gemv;
pub mod storage;

mod matrix;
mod operands;
#[allow(clippy::module_inception)]
mod strategy;

pub(crate) use matrix::*;
pub(crate) use operands::*;
pub use strategy::Strategy;

use crate::definition::MatmulSetupError;
use cubek_tile::launch::Refusal;

/// An operand the tile engine refused to bind is a configuration this routine cannot run.
impl From<Refusal> for MatmulSetupError {
    fn from(refusal: Refusal) -> Self {
        MatmulSetupError::InvalidConfig(Box::new(refusal))
    }
}

#[cfg(feature = "benchmarks")]
pub mod eval;
