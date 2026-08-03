use cubecl::features::Plane;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tf32;
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

/// Tunable knobs for [`BahtTsqrRoutine`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BahtTsqrStrategy {
    /// Trade accuracy for speed on the trailing-update GEMMs. See
    /// [`BahtTsqrBlueprint::allow_tf32`]. Off by default.
    pub allow_tf32: bool,
}

/// Comptime specialization settings for the TSQR kernels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BahtTsqrBlueprint {
    /// Fold the panel reductions with the hardware `plane_sum` (one shared
    /// slot per plane, two barriers) instead of the generic shared-memory
    /// tree. Requires plane-op support and a uniform plane size.
    pub use_plane_reduce: bool,
    /// Let the trailing-update GEMMs pick a tensor-core matmul, whose f32
    /// path rounds its inputs to tf32 (~10 mantissa bits instead of 24).
    ///
    /// Roughly **2x faster end to end** — those GEMMs are essentially the
    /// entire matmul cost — at a reconstruction error around `1e-2` rather
    /// than f32 accuracy. Off by default; enable it only when the caller's
    /// tolerance permits. No effect for f64, which has no tensor-core path.
    pub allow_tf32: bool,
}

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
        let hardware = &client.properties().hardware;

        let plane_reduce_supported = client.properties().features.plane.contains(Plane::Ops)
            && hardware.plane_size_min == hardware.plane_size_max
            && hardware.plane_size_min > 0;

        let tf32_supported = client
            .properties()
            .supports_type(tf32::as_type_native_unchecked().storage_type());

        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => {
                if blueprint.use_plane_reduce && !plane_reduce_supported {
                    return Err(QRSetupError::InvalidBlueprint(
                        "use_plane_reduce requires plane-op support and a uniform plane size"
                            .to_string(),
                    ));
                }
                blueprint
            }
            BlueprintStrategy::Inferred(strategy) => BahtTsqrBlueprint {
                use_plane_reduce: plane_reduce_supported,
                // Honour the opt-in only where tf32 actually exists. Without a
                // tensor-core path `Strategy::Auto` resolves to something
                // *slower* than the full-precision unit routine (measured on
                // wgpu: 24.1ms -> 30.5ms at 1024x1024), so taking the accuracy
                // hit there would buy nothing.
                allow_tf32: strategy.allow_tf32 && tf32_supported,
            },
        };
        let thread_block_size = (hardware.max_cube_dim.0 as f64).sqrt() as u32;
        let max_cube_dim = hardware.max_cube_dim.0.min(256);
        let tile = 32u32.min(problem.cols as u32).min(max_cube_dim);

        // `Strategy::Auto` resolves to a tensor-core (CMMA) matmul whose f32
        // path silently downgrades the stage/register types to tf32 (see
        // `cubek_matmul::definition::adjust_dtypes`), losing ~13 mantissa bits
        // and pushing the QR reconstruction error to ~1e-2 — over the test
        // tolerance — so f32 defaults to full-precision strategies.
        //
        // The unit routines don't support f64, so f64 uses Auto — safe there
        // because no f64 CMMA exists and Auto falls back to the
        // full-precision SimpleUnit.
        //
        // Only `strategy_tall` is worth trading accuracy for: measured at
        // 1024x1024, forcing it to Auto takes the whole decomposition from
        // 23.5ms to 12.0ms, while forcing gram or W to Auto changes nothing
        // end to end. So the tf32 opt-in touches the trailing updates alone
        // and gram/W stay full-precision unconditionally.
        let (strategy_gram, strategy_w, strategy_tall) = if is_f64(problem) {
            (Strategy::Auto, Strategy::Auto, Strategy::Auto)
        } else {
            let tall = if blueprint.allow_tf32 {
                Strategy::Auto
            } else {
                Strategy::DoubleUnit(Default::default())
            };
            (
                Strategy::SimpleVecMat(Default::default()),
                Strategy::DoubleUnit(Default::default()),
                tall,
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
