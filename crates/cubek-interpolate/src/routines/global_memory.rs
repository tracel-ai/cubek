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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalMemoryStrategy {
    pub tile_target_aspect_ratio: f32,
}

impl ForwardRoutine for GlobalMemoryRoutine {
    type Strategy = GlobalMemoryStrategy;
    type Blueprint = InterpolateBlueprint;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        strategy: BlueprintStrategy<Self>,
        vector_size: usize,
        _bytes_per_element: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let settings = prepare_global_launch_settings(client, problem, strategy, vector_size);

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
    strategy: BlueprintStrategy<GlobalMemoryRoutine>,
    vector_size: usize,
) -> InterpolateLaunchSettings {
    let num_vectors = problem.channels / vector_size;
    let working_units = problem.output_width * problem.output_height * num_vectors;

    let tile_target_aspect_ratio = match strategy {
        BlueprintStrategy::Forced(blueprint) => blueprint.tile_size.aspect_ratio(),
        BlueprintStrategy::Inferred(strategy) => strategy.tile_target_aspect_ratio,
    };

    let (cube_dim, tile_size, _) = compute_layout(
        client,
        working_units,
        num_vectors,
        tile_target_aspect_ratio,
        problem.options,
    );

    build_settings(
        client,
        problem,
        problem.options,
        cube_dim,
        tile_size,
        num_vectors,
    )
}
