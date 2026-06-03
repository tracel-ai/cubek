use crate::{
    InterpolateError,
    definition::InterpolateForwardProblem,
    routines::{
        BlueprintStrategy, ForwardRoutine, GlobalInterpolateBlueprint, GlobalMemoryBlueprint,
        InterpolateBlueprint, InterpolateLaunchSettings, build_settings, compute_layout,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryStrategy {}

impl ForwardRoutine for GlobalMemoryRoutine {
    type Strategy = GlobalMemoryStrategy;
    type Blueprint = GlobalMemoryBlueprint;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        _strategy: BlueprintStrategy<Self>,
        _bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let settings = prepare_global_launch_settings(client, problem, vector_size);

        let blueprint = InterpolateBlueprint {
            tile_size: settings.tile_size,
            options: problem.options,
            global: GlobalInterpolateBlueprint::GlobalMemoryBlueprint(GlobalMemoryBlueprint {}),
        };

        Ok((blueprint, settings))
    }
}

fn prepare_global_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    vector_size: usize,
) -> InterpolateLaunchSettings {
    let num_vectors = problem.channels / vector_size;
    let working_units = problem.output_width * problem.output_height * num_vectors;

    let (cube_dim, tile_size) = compute_layout(client, working_units, num_vectors, problem.options);

    build_settings(
        client,
        problem,
        problem.options,
        cube_dim,
        tile_size,
        num_vectors,
    )
}
