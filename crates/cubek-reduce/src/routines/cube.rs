use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
    ReduceVectorSettings,
};
use crate::{
    BoundChecks, IdleMode, ReduceError, VectorizationMode,
    launch::{calculate_plane_count_per_cube, support_plane},
    routines::{BlueprintStrategy, CubeBlueprint, Routine},
};
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, features::Plane};
use cubek_std::cube_count::cube_count_spread_with_total;

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
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let address_type = problem.address_type;
        let (blueprint, cube_dim, num_cubes) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                // One accumulator per plane.
                if blueprint.use_planes {
                    if !support_plane(client) {
                        return Err(ReduceError::PlanesUnavailable);
                    }

                    if blueprint.num_shared_accumulators != cube_dim.x as usize {
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
                } else if blueprint.num_shared_accumulators != cube_dim.num_elems() as usize {
                    return Err(ReduceError::Validation {
                        details: "Num accumulators should match cube_dim.num_elems()",
                    });
                }

                let working_cubes = working_cubes(&settings, &problem);
                let (cube_count, launched_cubes) =
                    cube_count_spread_with_total(client, working_cubes);

                if working_cubes != launched_cubes && !blueprint.cube_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many cubes launched for the problem causing OOD, but `cube_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    vectorization_mode: settings.vectorization_mode,
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
            address_type,
            vector: settings,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceVectorSettings,
    strategy: CubeStrategy,
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    if strategy.use_planes && !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let hardware_properties = &client.properties().hardware;
    let plane_size = hardware_properties.plane_size_max;

    let use_planes = strategy.use_planes
        && hardware_properties.plane_size_max == hardware_properties.plane_size_min;

    let working_cubes = working_cubes(settings, &problem);
    let working_units = working_cubes * problem.reduce_len.div_ceil(settings.vector_size_input);
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, hardware_properties);
    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let cube_size = cube_dim.num_elems();

    let work_size = match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_len / settings.vector_size_input,
        VectorizationMode::Perpendicular => problem.reduce_len,
    };
    let bound_checks = match work_size.is_multiple_of(cube_size as usize) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let num_shared_accumulators = match use_planes {
        true => plane_count as usize,
        false => cube_size as usize,
    };

    let (cube_count, launched_cubes) = cube_count_spread_with_total(client, working_cubes);

    let cube_idle = match working_cubes != launched_cubes {
        true => match strategy.use_planes
            && !client
                .properties()
                .features
                .plane
                .contains(Plane::NonUniformControlFlow)
        {
            true => IdleMode::Mask,
            false => IdleMode::Terminate,
        },
        false => IdleMode::None,
    };
    let blueprint = ReduceBlueprint {
        vectorization_mode: settings.vectorization_mode,
        global: GlobalReduceBlueprint::Cube(CubeBlueprint {
            cube_idle,
            bound_checks,
            num_shared_accumulators,
            use_planes,
        }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_cubes(settings: &ReduceVectorSettings, problem: &ReduceProblem) -> usize {
    match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_count / settings.vector_size_output,
        VectorizationMode::Perpendicular => problem.reduce_count / settings.vector_size_input,
    }
}
