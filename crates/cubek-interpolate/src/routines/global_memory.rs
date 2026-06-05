use crate::{
    InterpolateError,
    definition::{InterpolateForwardProblem, InterpolateMode, TileSize, get_transform},
    routines::{
        BlueprintStrategy, ForwardRoutine, GlobalInterpolateBlueprint, GlobalMemoryBlueprint,
        InterpolateBlueprint, InterpolateLaunchSettings, build_settings,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalMemoryStrategy {
    pub tile_size: TileSize,
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
        let tile_size = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_size,
            BlueprintStrategy::Inferred(strategy) => strategy.tile_size,
        };

        let is_flattened = is_flattened(problem);

        let settings =
            prepare_global_launch_settings(client, problem, tile_size, is_flattened, vector_size);

        let transform_width =
            get_transform(problem.input_width, problem.output_width, problem.options);
        let transform_height =
            get_transform(problem.input_height, problem.output_height, problem.options);

        let blueprint = InterpolateBlueprint {
            tile_size,
            options: problem.options,
            transform_width,
            transform_height,
            global: GlobalInterpolateBlueprint::GlobalMemoryBlueprint(GlobalMemoryBlueprint {
                is_flattened,
            }),
        };

        Ok((blueprint, settings))
    }
}

fn prepare_global_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    tile_size: TileSize,
    is_flattened: bool,
    vector_size: usize,
) -> InterpolateLaunchSettings {
    let num_vectors = problem.channels / vector_size;
    let working_units = problem.output_width * problem.output_height * num_vectors;

    let cube_dim = CubeDim::new(client, working_units);

    build_settings(
        client,
        problem,
        cube_dim,
        tile_size,
        is_flattened,
        num_vectors,
    )
}

#[allow(clippy::match_like_matches_macro)]
fn is_flattened(problem: &InterpolateForwardProblem) -> bool {
    match problem.options.mode {
        InterpolateMode::Nearest(_) => true,
        _ => false,
    }
}
