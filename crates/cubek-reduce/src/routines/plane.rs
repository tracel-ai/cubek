use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    BoundChecks, LineMode, ReduceError,
    launch::{calculate_plane_count_per_cube, support_plane},
    routines::{BlueprintStrategy, PlaneReduceBlueprint, Routine, cube_count_safe},
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
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, cube_count) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                if !support_plane(client) {
                    return Err(ReduceError::PlanesUnavailable);
                }

                if cube_dim.x != client.properties().hardware.plane_size_max {
                    return Err(ReduceError::Validation {
                        details: "`cube_dim.x` must match `plane_size_max`",
                    });
                }

                let working_planes = working_planes(&settings, &problem);

                let working_cubes = working_planes.div_ceil(cube_dim.y);
                let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);
                let plane_idle = launched_cubes * cube_dim.y != working_planes;

                if plane_idle && !blueprint.plane_idle {
                    return Err(ReduceError::Validation {
                        details: "Too many planes launched for the problem causing OOD, but `plane_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::Plane(blueprint),
                };

                (blueprint, cube_dim, cube_count)
            }
            BlueprintStrategy::Inferred(strategy) => {
                let (blueprint, cube_dim, cube_count) =
                    generate_blueprint::<R>(client, problem, &settings, strategy)?;
                (blueprint, cube_dim, cube_count)
            }
        };

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
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    if !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_planes = working_planes(settings, &problem);
    let working_units = working_planes * plane_size;
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, properties.num_cpu_cores);
    let working_cubes = working_planes.div_ceil(plane_count);

    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let (cube_count, cube_launched) = cube_count_safe(client, working_cubes);

    let plane_idle = cube_launched * cube_dim.num_elems() != working_units;
    let work_size = match settings.line_mode {
        LineMode::Parallel => problem.vector_size / settings.line_size_input as u32,
        LineMode::Perpendicular => problem.vector_size,
    };
    let bound_checks = match work_size.is_multiple_of(plane_size) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let blueprint = ReduceBlueprint {
        line_mode: settings.line_mode,
        global: GlobalReduceBlueprint::Plane(PlaneReduceBlueprint {
            plane_idle,
            bound_checks,
            independent: strategy.independent,
        }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_planes(settings: &ReduceLineSettings, problem: &ReduceProblem) -> u32 {
    match settings.line_mode {
        LineMode::Parallel => problem.vector_count / settings.line_size_output as u32,
        LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
    }
}
