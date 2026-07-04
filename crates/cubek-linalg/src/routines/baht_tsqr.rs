use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubek_matmul::strategy::Strategy;

use crate::{
    definition::{QRProblem, QRSetupError},
    routines::{BlueprintStrategy, QRRoutine},
};

/// Whether the element type is f64, for which the specialized unit matmul
/// routines are not supported; fall back to auto-selection in that case.
fn is_f64(problem: &QRProblem) -> bool {
    problem.dtype == StorageType::Scalar(ElemType::Float(FloatKind::F64))
}

/// TSQR-inspired blocked Householder QR: the whole panel is factorized with
/// minimal dispatches, then the block reflector is applied through GEMMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BahtTsqrRoutine;

/// Tunable knobs for [`BahtTsqrRoutine`]. There are none yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BahtTsqrStrategy;

/// The TSQR kernels have no comptime specialization settings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BahtTsqrBlueprint;

/// Runtime launch parameters for the TSQR kernels and GEMM updates.
#[derive(Clone)]
pub struct BahtTsqrLaunchSettings {
    /// Number of reflectors batched per panel.
    pub tile: u32,
    /// Cube dim used for the 1D panel kernels.
    pub max_cube_dim: u32,
    /// Side of the square cube dim used for the 2D update kernels.
    pub thread_block_size: u32,
    pub cube_dim_2d: CubeDim,
    /// Matmul strategy for the V^T·V Gram matrix.
    pub strategy_gram: Strategy,
    /// Matmul strategy for W = V·T.
    pub strategy_w: Strategy,
    /// Matmul strategy for the tall trailing-R and Q^T updates.
    pub strategy_tall: Strategy,
}

impl QRRoutine for BahtTsqrRoutine {
    type Strategy = BahtTsqrStrategy;
    type Blueprint = BahtTsqrBlueprint;
    type LaunchSettings = BahtTsqrLaunchSettings;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &QRProblem,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(Self::Blueprint, Self::LaunchSettings), QRSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint,
            BlueprintStrategy::Inferred(_) => BahtTsqrBlueprint,
        };

        let hardware = &client.properties().hardware;
        let thread_block_size = (hardware.max_cube_dim.0 as f64).sqrt() as u32;
        let max_cube_dim = hardware.max_cube_dim.0.min(256);
        let tile = 32u32.min(problem.cols as u32).min(max_cube_dim);

        let (strategy_gram, strategy_w, strategy_tall) = if is_f64(problem) {
            (Strategy::Auto, Strategy::Auto, Strategy::Auto)
        } else {
            (
                Strategy::SimpleVecMat(Default::default()),
                Strategy::DoubleUnit(Default::default()),
                Strategy::SimpleUnit(Default::default()),
            )
        };

        let settings = BahtTsqrLaunchSettings {
            tile,
            max_cube_dim,
            thread_block_size,
            cube_dim_2d: CubeDim::new_2d(thread_block_size, thread_block_size),
            strategy_gram,
            strategy_w,
            strategy_tall,
        };

        Ok((blueprint, settings))
    }
}
