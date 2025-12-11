use super::{GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings};
use crate::{
    BoundChecks, LineMode, ReduceError,
    launch::{calculate_plane_count, support_plane},
    routines::{CubeReduceBlueprint, Routine},
};
use cubecl::{CubeCount, CubeDim, Runtime};

pub struct CubeRoutine;

#[derive(Debug, Clone)]
pub struct CubeStrategy {
    /// If we use plane to aggregate accumulators.
    pub use_planes: bool,
}

impl<R: Runtime> Routine<R> for CubeRoutine {
    type Strategy = CubeStrategy;

    fn prepare(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        problem: super::ReduceProblem,
        settings: super::ReduceLineSettings,
        strategy: Self::Strategy,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        if strategy.use_planes && !support_plane(client) {
            return Err(ReduceError::PlanesUnavailable);
        }

        let properties = &client.properties().hardware;
        let plane_size = properties.plane_size_max;
        let working_cubes = match settings.line_mode {
            LineMode::Parallel => problem.vector_count,
            LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
        };
        let plane_count = calculate_plane_count(
            working_cubes * problem.vector_size,
            plane_size,
            properties.num_cpu_cores,
        );
        let cube_dim = CubeDim::new_2d(plane_size, plane_count);
        let cube_size = cube_dim.num_elems();

        let bound_checks = match working_cubes % cube_size != 0 {
            true => BoundChecks::Mask,
            false => BoundChecks::None,
        };

        let num_shared_accumulators = match strategy.use_planes {
            true => plane_count,
            false => cube_size,
        };

        let blueprint = ReduceBlueprint {
            line_mode: settings.line_mode,
            global: GlobalReduceBlueprint::Cube(CubeReduceBlueprint {
                bound_checks,
                num_shared_accumulators,
                use_planes: strategy.use_planes,
            }),
        };

        let cube_count = CubeCount::new_1d(working_cubes);
        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count,
            line: settings,
        };

        Ok((blueprint, launch))
    }
}
