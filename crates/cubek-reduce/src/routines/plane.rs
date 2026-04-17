use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
    ReduceVectorSettings,
};
use crate::{
    BoundChecks, IdleMode, ReduceError, VectorizationMode,
    launch::{calculate_plane_count_per_cube, support_plane},
    routines::{BlueprintStrategy, PlaneMergeStrategy, PlaneReduceBlueprint, Routine},
};
use cubecl::{CubeCount, CubeDim, Runtime, features::Plane, prelude::ComputeClient};
use cubek_std::cube_count::cube_count_spread_with_total;

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
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let address_type = problem.address_type;
        let (blueprint, cube_dim, cube_count) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                if !support_plane(client) {
                    return Err(ReduceError::PlanesUnavailable);
                }

                let properties = &client.properties().hardware;
                if cube_dim.x != properties.plane_size_max {
                    return Err(ReduceError::Validation {
                        details: "`cube_dim.x` must match `plane_size_max`",
                    });
                }

                let working_planes = working_planes(&settings, &problem);

                let working_cubes = working_planes.div_ceil(cube_dim.y as usize);
                let (cube_count, launched_cubes) =
                    cube_count_spread_with_total(client, working_cubes);
                let plane_idle = launched_cubes * cube_dim.y as usize != working_planes;

                if plane_idle && !blueprint.plane_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many planes launched for the problem causing OOD, but `plane_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    vectorization_mode: settings.vectorization_mode,
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
    strategy: PlaneStrategy,
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    if !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_planes = working_planes(settings, &problem);
    let working_units = working_planes * plane_size as usize;
    let plane_count = calculate_plane_count_per_cube(working_units, plane_size, properties);
    let working_cubes = working_planes.div_ceil(plane_count as usize);

    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let (cube_count, cube_launched) = cube_count_spread_with_total(client, working_cubes);

    let plane_idle = cube_launched * cube_dim.num_elems() as usize != working_units;
    let work_size = match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.vector_size / settings.vector_size_input,
        VectorizationMode::Perpendicular => problem.vector_size,
    };
    let bound_checks = match work_size.is_multiple_of(plane_size as usize) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let plane_idle = match plane_idle {
        true => match client
            .properties()
            .features
            .plane
            .contains(Plane::NonUniformControlFlow)
        {
            true => IdleMode::Terminate,
            false => IdleMode::Mask,
        },
        false => IdleMode::None,
    };

    let strategy_enum = if strategy.independent {
        PlaneMergeStrategy::Lazy
    } else {
        PlaneMergeStrategy::Eager
    };

    let blueprint = ReduceBlueprint {
        vectorization_mode: settings.vectorization_mode,
        global: GlobalReduceBlueprint::Plane(PlaneReduceBlueprint {
            plane_idle,
            bound_checks,
            plane_merge_strategy: strategy_enum,
            plane_dim_ceil: properties.plane_size_max != properties.plane_size_min,
        }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_planes(settings: &ReduceVectorSettings, problem: &ReduceProblem) -> usize {
    match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.vector_count / settings.vector_size_output,
        VectorizationMode::Perpendicular => problem.vector_count / settings.vector_size_input,
    }
}
