use cubecl::prelude::*;

use crate::{
    definition::{QRProblem, QRSetupError},
    routines::{BlueprintStrategy, QRRoutine},
};

/// Modified Gram-Schmidt: a single persistent cube processes one column per
/// launch, keeping the running column vector in shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MgsRoutine;

/// Tunable knobs for [`MgsRoutine`]. There are none yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MgsStrategy;

/// Comptime settings for the MGS column kernel.
///
/// Both lengths are shared-memory slice capacities, which must be known at
/// compile time. They are rounded to powers of two so that nearby problem
/// sizes reuse the same compiled kernel instead of retriggering JIT
/// compilation for every distinct row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MgsBlueprint {
    /// Capacity (in elements) of the shared working column vector; at least `rows`.
    pub v_len: usize,
    /// Slots of the shared tree-reduction buffer; equals the cube dim along x.
    pub reduce_len: usize,
}

/// Runtime launch parameters for the MGS column kernel.
#[derive(Debug, Clone)]
pub struct MgsLaunchSettings {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

impl QRRoutine for MgsRoutine {
    type Strategy = MgsStrategy;
    type Blueprint = MgsBlueprint;
    type LaunchSettings = MgsLaunchSettings;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &QRProblem,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(Self::Blueprint, Self::LaunchSettings), QRSetupError> {
        let hardware = &client.properties().hardware;
        let max_cube_dim = hardware.max_cube_dim.0 as usize;

        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => {
                if blueprint.v_len < problem.rows {
                    return Err(QRSetupError::InvalidBlueprint(format!(
                        "v_len ({}) must hold a full column ({} rows)",
                        blueprint.v_len, problem.rows
                    )));
                }
                if blueprint.reduce_len > max_cube_dim {
                    return Err(QRSetupError::InvalidBlueprint(format!(
                        "reduce_len ({}) exceeds the maximum cube dim ({max_cube_dim})",
                        blueprint.reduce_len
                    )));
                }
                blueprint
            }
            BlueprintStrategy::Inferred(_) => {
                let cube_dim_x = problem.rows.next_power_of_two().clamp(1, max_cube_dim);
                MgsBlueprint {
                    v_len: problem.rows.next_power_of_two().max(1),
                    reduce_len: cube_dim_x,
                }
            }
        };

        let requested = (blueprint.v_len + blueprint.reduce_len) * problem.dtype.size();
        let available = hardware.max_shared_memory_size;
        if requested > available {
            return Err(QRSetupError::SharedMemoryLimitExceeded {
                requested,
                available,
            });
        }

        let settings = MgsLaunchSettings {
            cube_dim: CubeDim::new_1d(blueprint.reduce_len as u32),
            cube_count: CubeCount::new_1d(1),
        };

        Ok((blueprint, settings))
    }
}
