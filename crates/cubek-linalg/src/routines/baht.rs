use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubek_matmul::strategy::Strategy;

use crate::{
    definition::{QRProblem, QRSetupError},
    routines::{BlueprintStrategy, QRRoutine},
};

/// Blocked Householder QR: batches reflectors with the compact WY
/// representation so the trailing updates run as large GEMMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BahtRoutine;

/// Tunable knobs for [`BahtRoutine`]. There are none yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BahtStrategy;

/// Comptime settings for the BAHT panel kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BahtBlueprint {
    /// Capacity (in elements) of the shared buffers used by the panel
    /// kernels; equals the cube dim the panel kernels are launched with.
    pub shared_size: usize,
}

/// Runtime launch parameters for the BAHT kernels and GEMM updates.
#[derive(Clone)]
pub struct BahtLaunchSettings {
    /// Number of reflectors batched per panel, bounded by shared memory.
    pub tile: u32,
    /// Cube dim used for the 1D panel kernels.
    pub max_cube_dim: u32,
    /// Side of the square cube dim used for the 2D update kernels.
    pub thread_block_size: u32,
    pub cube_dim_2d: CubeDim,
    /// Matmul strategy for the trailing-R and Q^T GEMM updates.
    pub matmul_strategy: Strategy,
}

/// Whether the element type is f64, for which the specialized unit matmul
/// routines are not supported; fall back to auto-selection in that case.
pub(crate) fn is_f64(problem: &QRProblem) -> bool {
    problem.dtype == StorageType::Scalar(ElemType::Float(FloatKind::F64))
}

impl QRRoutine for BahtRoutine {
    type Strategy = BahtStrategy;
    type Blueprint = BahtBlueprint;
    type LaunchSettings = BahtLaunchSettings;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &QRProblem,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(Self::Blueprint, Self::LaunchSettings), QRSetupError> {
        let hardware = &client.properties().hardware;
        let hw_max_cube_dim = hardware.max_cube_dim.0;
        let thread_block_size = (hw_max_cube_dim as f64).sqrt() as u32;

        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => {
                if blueprint.shared_size == 0 || blueprint.shared_size > hw_max_cube_dim as usize {
                    return Err(QRSetupError::InvalidBlueprint(format!(
                        "shared_size ({}) must be in 1..={hw_max_cube_dim}",
                        blueprint.shared_size
                    )));
                }
                blueprint
            }
            BlueprintStrategy::Inferred(_) => BahtBlueprint {
                shared_size: hw_max_cube_dim.min(256) as usize,
            },
        };

        // The panel factorization stores V and W ([rows, tile] each) built
        // incrementally in global memory, but the tile width is bounded so the
        // per-tile working set stays comparable to the shared memory budget.
        let bytes_per_elem = problem.dtype.size();
        let max_tile_from_shared =
            ((hardware.max_shared_memory_size / bytes_per_elem) as f64).sqrt() as u32;
        let mut tile = 1u32;
        while tile * 2 <= max_tile_from_shared {
            tile *= 2;
        }
        let tile = tile.min(problem.cols as u32);

        let matmul_strategy = if is_f64(problem) {
            Strategy::Auto
        } else {
            Strategy::SimpleUnit(Default::default())
        };

        let settings = BahtLaunchSettings {
            tile,
            max_cube_dim: blueprint.shared_size as u32,
            thread_block_size,
            cube_dim_2d: CubeDim::new_2d(thread_block_size, thread_block_size),
            matmul_strategy,
        };

        Ok((blueprint, settings))
    }
}
