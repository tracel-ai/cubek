use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    definition::{QRProblem, QRSetupError},
    routines::{BlueprintStrategy, QRRoutine},
};

/// Givens rotations: one parallel rotation pass per column, better suited for
/// sparse matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CgrRoutine;

/// Tunable knobs for [`CgrRoutine`]. There are none yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CgrStrategy;

/// The CGR kernels have no comptime specialization settings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CgrBlueprint;

/// Runtime launch parameters for the CGR kernels: one working unit per row.
#[derive(Debug, Clone)]
pub struct CgrLaunchSettings {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

impl QRRoutine for CgrRoutine {
    type Strategy = CgrStrategy;
    type Blueprint = CgrBlueprint;
    type LaunchSettings = CgrLaunchSettings;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &QRProblem,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(Self::Blueprint, Self::LaunchSettings), QRSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint,
            BlueprintStrategy::Inferred(_) => CgrBlueprint,
        };

        let max_cube_dim = client.properties().hardware.max_cube_dim.0;
        let cube_dim = CubeDim::new_1d((problem.rows as u32).min(max_cube_dim));
        let cube_count = calculate_cube_count_elemwise(client, problem.rows, cube_dim);

        Ok((blueprint, CgrLaunchSettings {
            cube_dim,
            cube_count,
        }))
    }
}
