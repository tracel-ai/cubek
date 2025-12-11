use super::{GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings};
use crate::{
    BoundChecks, LineMode, ReduceError,
    launch::{calculate_plane_count, support_plane},
    routines::{PlaneReduceBlueprint, Routine},
};
use cubecl::{CubeCount, CubeDim, Runtime};

pub struct PlaneRoutine;

#[derive(Debug, Clone)]
pub struct PlaneStrategy {
    /// How the accumulators are handled in a plane.
    pub independant: bool,
}

impl<R: Runtime> Routine<R> for PlaneRoutine {
    type Strategy = PlaneStrategy;

    fn prepare(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        problem: super::ReduceProblem,
        settings: super::ReduceLineSettings,
        strategy: Self::Strategy,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        if !support_plane(client) {
            return Err(ReduceError::PlanesUnavailable);
        }

        let properties = &client.properties().hardware;
        let plane_size = properties.plane_size_max;
        let working_planes = match settings.line_mode {
            LineMode::Parallel => problem.vector_count,
            LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
        };
        let working_units = working_planes * plane_size;

        let plane_count =
            calculate_plane_count(working_units, plane_size, properties.num_cpu_cores);

        let cube_dim = CubeDim::new_2d(plane_size, plane_count);
        let plane_idle = working_planes % plane_count != 0;
        let bound_checks = match problem.vector_size % plane_size != 0 {
            true => BoundChecks::Mask,
            false => BoundChecks::None,
        };

        let blueprint = ReduceBlueprint {
            line_mode: settings.line_mode,
            global: GlobalReduceBlueprint::FullPlane(PlaneReduceBlueprint {
                plane_idle,
                bound_checks,
                independant: strategy.independant,
            }),
        };

        let cube_count = working_planes.div_ceil(plane_count);
        let cube_count = CubeCount::new_1d(cube_count);
        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count,
            line: settings,
        };

        Ok((blueprint, launch))
    }
}
