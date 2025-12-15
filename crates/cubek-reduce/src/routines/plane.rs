use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    BoundChecks, LineMode, ReduceError,
    launch::{calculate_plane_count, support_plane},
    routines::{PlaneReduceBlueprint, Routine, RoutineStrategy},
};
use cubecl::{CubeCount, CubeDim, Runtime, prelude::ComputeClient};

#[derive(Debug, Clone)]
pub struct PlaneRoutine;

#[derive(Debug, Clone)]
pub struct PlaneStrategy {
    /// How the accumulators are handled in a plane.
    pub independent: bool,
}

impl Routine for PlaneRoutine {
    type Strategy = PlaneStrategy;
    type Blueprint = PlaneReduceBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: RoutineStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, working_planes) = match strategy {
            RoutineStrategy::Forced(blueprint, cube_dim) => {
                if !support_plane(client) {
                    return Err(ReduceError::PlanesUnavailable);
                }

                if cube_dim.x != client.properties().hardware.plane_size_max {
                    return Err(ReduceError::Validation {
                        details: "`cube_dim.x` must match `plane_size_max`",
                    });
                }

                let working_planes = working_planes(&settings, &problem);
                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::FullPlane(blueprint),
                };

                (blueprint, cube_dim, working_planes)
            }
            RoutineStrategy::Strategy(strategy) => {
                let (blueprint, cube_dim, working_planes) =
                    generate_blueprint::<R>(client, problem, &settings, strategy)?;
                (blueprint, cube_dim, working_planes)
            }
        };

        let cube_count = working_planes.div_ceil(cube_dim.y);
        let cube_count = CubeCount::new_1d(cube_count);
        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count,
            line: settings,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceLineSettings,
    strategy: PlaneStrategy,
) -> Result<(ReduceBlueprint, CubeDim, u32), ReduceError> {
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

    let plane_count = calculate_plane_count(working_units, plane_size, properties.num_cpu_cores);

    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let plane_idle = working_planes % plane_count != 0;
    let bound_checks = match !problem.vector_size.is_multiple_of(plane_size) {
        true => BoundChecks::Mask,
        false => BoundChecks::None,
    };

    let blueprint = ReduceBlueprint {
        line_mode: settings.line_mode,
        global: GlobalReduceBlueprint::FullPlane(PlaneReduceBlueprint {
            plane_idle,
            bound_checks,
            independent: strategy.independent,
        }),
    };

    Ok((blueprint, cube_dim, working_planes))
}

fn working_planes(settings: &ReduceLineSettings, problem: &ReduceProblem) -> u32 {
    match settings.line_mode {
        LineMode::Parallel => problem.vector_count,
        LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
    }
}
