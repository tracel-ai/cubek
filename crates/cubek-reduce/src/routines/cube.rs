use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    BoundChecks, LineMode, ReduceError,
    launch::{calculate_plane_count_per_cube, support_plane},
    routines::{BlueprintStrategy, CubeBlueprint, Routine, cube_count_safe},
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
    type Blueprint = CubeBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, num_cubes) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
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
                let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);

                if working_cubes != launched_cubes && !blueprint.cube_idle {
                    return Err(ReduceError::Validation {
                        details: "Too many cubes launched for the problem causing OOD, but `cube_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::Cube(blueprint),
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
            cube_count: num_cubes,
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
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    if strategy.use_planes && !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_cubes = working_cubes(settings, &problem);
    let working_units = working_cubes
        * problem
            .vector_size
            .div_ceil(settings.line_size_input as u32);
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, properties.num_cpu_cores);
    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let cube_size = cube_dim.num_elems();

    let work_size = match settings.line_mode {
        LineMode::Parallel => problem.vector_size / settings.line_size_input as u32,
        LineMode::Perpendicular => problem.vector_size,
    };
    let bound_checks = match work_size.is_multiple_of(cube_size) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let num_shared_accumulators = match strategy.use_planes {
        true => plane_count,
        false => cube_size,
    };

    let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);

    let cube_idle = working_cubes != launched_cubes;
    let blueprint = ReduceBlueprint {
        line_mode: settings.line_mode,
        global: GlobalReduceBlueprint::Cube(CubeBlueprint {
            cube_idle,
            bound_checks,
            num_shared_accumulators,
            use_planes: strategy.use_planes,
        }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_cubes(settings: &ReduceLineSettings, problem: &ReduceProblem) -> u32 {
    match settings.line_mode {
        LineMode::Parallel => problem.vector_count / settings.line_size_output as u32,
        LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
    }
}
