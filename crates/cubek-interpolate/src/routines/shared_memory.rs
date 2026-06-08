use crate::{
    InterpolateError,
    definition::{
        InterpolateForwardProblem, InterpolateOptions, TileSize, Transform, get_halo, get_transform,
    },
    routines::{
        BlueprintStrategy, ForwardRoutine, GlobalInterpolateBlueprint, InterpolateBlueprint,
        InterpolateLaunchSettings, SharedMemoryBlueprint, build_settings,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedMemoryStrategy {
    pub tile_size: TileSize,
}

impl ForwardRoutine for SharedMemoryRoutine {
    type Strategy = SharedMemoryStrategy;
    type Blueprint = InterpolateBlueprint;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        strategy: BlueprintStrategy<Self>,
        vector_size: usize,
        bytes_per_element: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let tile_size = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_size,
            BlueprintStrategy::Inferred(strategy) => strategy.tile_size,
        };

        let transform_width =
            get_transform(problem.input_width, problem.output_width, problem.options);
        let transform_height =
            get_transform(problem.input_height, problem.output_height, problem.options);

        let (settings, smem_width, smem_height) = prepare_shared_launch_settings(
            client,
            problem,
            tile_size,
            transform_width,
            transform_height,
            bytes_per_element,
            vector_size,
            client.properties().hardware.max_shared_memory_size,
        )?;

        let blueprint = InterpolateBlueprint {
            tile_size,
            options: problem.options,
            transform_width,
            transform_height,
            global: GlobalInterpolateBlueprint::SharedMemoryBlueprint(SharedMemoryBlueprint {
                smem_width,
                smem_height,
                num_vectors: settings.num_vectors,
            }),
        };

        Ok((blueprint, settings))
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_shared_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    tile_size: TileSize,
    transform_width: Transform,
    transform_height: Transform,
    bytes_per_element: usize,
    vector_size: usize,
    max_shared_memory_bytes: usize,
) -> Result<(InterpolateLaunchSettings, usize, usize), InterpolateError> {
    let num_vectors = problem.channels / vector_size;
    let working_units = problem.output_width * problem.output_height * num_vectors;

    let cube_dim = CubeDim::new(client, working_units);

    let (smem_width, smem_height) = compute_smem_size(
        problem,
        problem.options,
        tile_size,
        transform_width,
        transform_height,
    );

    let requested_smem_bytes = smem_width * smem_height * num_vectors * bytes_per_element;

    // Check if the requested shared memory size fits within the hardware limits.
    if requested_smem_bytes <= max_shared_memory_bytes {
        let settings = build_settings(client, problem, cube_dim, tile_size, false, num_vectors);
        Ok((settings, smem_width, smem_height))
    } else {
        Err(InterpolateError::SharedMemoryLimitExceeded {
            requested: requested_smem_bytes,
            available: max_shared_memory_bytes,
        })
    }
}

fn compute_smem_size(
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    tile_size: TileSize,
    transform_width: Transform,
    transform_height: Transform,
) -> (usize, usize) {
    // Compute the effective tile footprint in output space.
    let (effective_width, effective_height) = (
        tile_size.width().min(problem.output_width),
        tile_size.height().min(problem.output_height),
    );

    // Calculate the scale factor for the distance this tile covers in the input image.
    let scale_x = get_span_scale(transform_width);
    let scale_y = get_span_scale(transform_height);

    let span_width = (effective_width.saturating_sub(1) as f64) * scale_x;
    let span_height = (effective_height.saturating_sub(1) as f64) * scale_y;

    // Add halo required by the specific interpolation mode.
    let halo = get_halo(options.mode);
    let smem_width = span_width.ceil() as usize + halo;
    let smem_height = span_height.ceil() as usize + halo;

    (smem_width.max(1), smem_height.max(1))
}

fn get_span_scale(transform: Transform) -> f64 {
    (transform.scale_numerator as f64) / (transform.scale_denominator as f64)
}
