use super::{GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings};
use crate::{
    LineMode, ReduceError,
    launch::calculate_plane_count,
    routines::{Routine, RoutineStrategy, UnitReduceBlueprint},
};
use cubecl::{CubeCount, CubeDim, Runtime};

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
        problem: super::ReduceProblem,
        settings: super::ReduceLineSettings,
        _strategy: RoutineStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let properties = &client.properties().hardware;
        let plane_size = properties.plane_size_max;
        let working_units = match settings.line_mode {
            LineMode::Parallel => problem.vector_count,
            LineMode::Perpendicular => problem.vector_count / settings.line_size_input as u32,
        };
        let plane_count =
            calculate_plane_count(working_units, plane_size, properties.num_cpu_cores);

        let cube_dim = CubeDim::new_2d(plane_size, plane_count);
        let num_units_in_cube = cube_dim.num_elems();
        let unit_idle = working_units % num_units_in_cube != 0;

        let blueprint = ReduceBlueprint {
            line_mode: settings.line_mode,
            global: GlobalReduceBlueprint::FullUnit(UnitReduceBlueprint { unit_idle }),
        };

        let cube_count = working_units.div_ceil(num_units_in_cube);
        let cube_count = match plane_size {
            // CPU
            1 => CubeCount::new_2d(1, cube_count),
            // GPU
            _ => CubeCount::new_1d(cube_count),
        };
        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count,
            line: settings,
        };

        Ok((blueprint, launch))
    }
}
