use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    BoundChecks, LineMode, ReduceError,
    launch::{calculate_plane_count, support_plane},
    routines::{CubeReduceBlueprint, Routine, RoutineStrategy},
};
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient};

#[derive(Debug, Clone)]
pub struct CubeRoutine;

#[derive(Debug, Clone)]
pub struct CubeStrategy {
    /// If we use plane to aggregate accumulators.
    pub use_planes: bool,
}

impl Routine for CubeRoutine {
    type Strategy = CubeStrategy;
    type Blueprint = CubeReduceBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: RoutineStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, cube_count) = match strategy {
            RoutineStrategy::Forced(blueprint, cube_dim) => {
                // One accumulator per plane.
                if blueprint.use_planes {
                    if !support_plane(client) {
                        return Err(ReduceError::PlanesUnavailable);
                    }

                    if blueprint.num_shared_accumulators != cube_dim.x {
                        return Err(ReduceError::Validation {
                            details: "Num accumulators should match cube_dim.x",
                        });
                    }
                    if cube_dim.x != client.properties().hardware.plane_size_max {
                        return Err(ReduceError::Validation {
                            details: "`cube_dim.x` must match `plane_size_max`",
                        });
                    }
                // One accumulator per unit.
                } else if blueprint.num_shared_accumulators != cube_dim.num_elems() {
                    return Err(ReduceError::Validation {
                        details: "Num accumulators should match cube_dim.num_elems()",
                    });
                }

                let working_cubes = working_cubes(&settings, &problem);
                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::Cube(blueprint),
                };
                (blueprint, cube_dim, CubeCount::new_1d(working_cubes))
            }
            RoutineStrategy::Strategy(strategy) => {
                let (blueprint, cube_dim, working_cubes) =
                    generate_blueprint::<R>(client, problem, &settings, strategy)?;
                (blueprint, cube_dim, CubeCount::new_1d(working_cubes))
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
    strategy: CubeStrategy,
) -> Result<(ReduceBlueprint, CubeDim, u32), ReduceError> {
    if strategy.use_planes && !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_cubes = working_cubes(settings, &problem);
    let plane_count = calculate_plane_count(
        working_cubes * problem.vector_size,
        plane_size,
        properties.num_cpu_cores,
    );
    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let cube_size = cube_dim.num_elems();

    let bound_checks = match !problem.vector_size.is_multiple_of(cube_size) {
        true => BoundChecks::Mask,
        false => BoundChecks::Mask, // TODO
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

    Ok((blueprint, cube_dim, working_cubes))
}

fn working_cubes(settings: &ReduceLineSettings, problem: &ReduceProblem) -> u32 {
    match settings.line_mode {
        LineMode::Parallel => problem.vector_count,
        LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
    }
}
