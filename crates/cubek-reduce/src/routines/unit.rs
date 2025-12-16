use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    LineMode, ReduceError,
    launch::calculate_plane_count_per_cube,
    routines::{BlueprintStrategy, Routine, UnitReduceBlueprint, cube_count_safe},
};
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient};

#[derive(Debug, Clone)]
pub struct UnitRoutine;

#[derive(Debug, Clone)]
pub struct UnitStrategy;

impl Routine for UnitRoutine {
    type Strategy = UnitStrategy;
    type Blueprint = UnitReduceBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, cube_count) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                let working_units = working_units(&settings, &problem);
                let num_units_in_cube = cube_dim.num_elems();
                let working_cubes = working_units.div_ceil(num_units_in_cube);

                let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);

                if working_cubes != launched_cubes && blueprint.unit_idle {
                    return Err(ReduceError::Validation {
                        details: "Too many units launched for the problem causing OOD, but `unit_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::Unit(blueprint),
                };

                (blueprint, cube_dim, cube_count)
            }
            BlueprintStrategy::Inferred(_) => {
                let (blueprint, cube_dim, cube_count) =
                    generate_blueprint::<R>(client, problem, &settings)?;
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
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_units = working_units(settings, &problem);
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, properties.num_cpu_cores);

    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let num_units_in_cube = cube_dim.num_elems();

    let working_cubes = working_units.div_ceil(num_units_in_cube);
    let (cube_count, cube_launched) = cube_count_safe(client, working_cubes);
    let unit_idle =
        !working_units.is_multiple_of(num_units_in_cube) || cube_launched != working_cubes;

    let blueprint = ReduceBlueprint {
        line_mode: settings.line_mode,
        global: GlobalReduceBlueprint::Unit(UnitReduceBlueprint { unit_idle }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_units(settings: &ReduceLineSettings, problem: &ReduceProblem) -> u32 {
    match settings.line_mode {
        LineMode::Parallel => problem.vector_count / settings.line_size_output as u32,
        LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
    }
}
