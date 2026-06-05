use crate::{
    InterpolateError,
    definition::{InterpolateForwardProblem, InterpolateOptions, TileSize, get_halo},
    routines::{
        BlueprintStrategy, ForwardRoutine, GlobalInterpolateBlueprint, InterpolateBlueprint,
        InterpolateLaunchSettings, SharedMemoryBlueprint, build_settings, compute_layout,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedMemoryStrategy {
    pub tile_target_aspect_ratio: f32,
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
        let (settings, smem_width, smem_height) = prepare_shared_launch_settings(
            client,
            problem,
            strategy,
            bytes_per_element,
            vector_size,
            client.properties().hardware.max_shared_memory_size,
        )?;

        let blueprint = InterpolateBlueprint {
            tile_size: settings.tile_size,
            options: problem.options,
            global: GlobalInterpolateBlueprint::SharedMemoryBlueprint(SharedMemoryBlueprint {
                smem_width,
                smem_height,
                num_vectors: settings.num_vectors,
            }),
        };

        Ok((blueprint, settings))
    }
}

fn prepare_shared_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    strategy: BlueprintStrategy<SharedMemoryRoutine>,
    bytes_per_element: usize,
    vector_size: usize,
    max_shared_memory_bytes: usize,
) -> Result<(InterpolateLaunchSettings, usize, usize), InterpolateError> {
    let num_vectors = problem.channels / vector_size;
    let mut working_units = problem.output_width * problem.output_height * num_vectors;

    let tile_target_aspect_ratio = match strategy {
        BlueprintStrategy::Forced(blueprint) => blueprint.tile_size.aspect_ratio(),
        BlueprintStrategy::Inferred(strategy) => strategy.tile_target_aspect_ratio,
    };

    loop {
        let (cube_dim, tile_size, units_per_cube) = compute_layout(
            client,
            working_units,
            tile_target_aspect_ratio,
            problem.options,
        );
        let (smem_width, smem_height) = compute_smem_size(problem, problem.options, tile_size);

        let requested_smem_bytes = smem_width * smem_height * num_vectors * bytes_per_element;

        // Check if the requested shared memory size fits within the hardware limits.
        if requested_smem_bytes <= max_shared_memory_bytes {
            let settings = build_settings(
                client,
                problem,
                problem.options,
                cube_dim,
                tile_size,
                num_vectors,
            );
            return Ok((settings, smem_width, smem_height));
        } else {
            if working_units <= 1 {
                return Err(InterpolateError::SharedMemoryLimitExceeded {
                    requested: requested_smem_bytes,
                    available: max_shared_memory_bytes,
                });
            }

            // Reduce the total units by half and try again.
            working_units = (units_per_cube / 2).max(1);
        }
    }
}

fn compute_smem_size(
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    tile_size: TileSize,
) -> (usize, usize) {
    // Calculate scaling factors between input and output dimensions.
    let scale_width = problem.input_width as f64 / problem.output_width as f64;
    let scale_height = problem.input_height as f64 / problem.output_height as f64;

    // Determine how many output rows a flattened 1D tile spans.
    let total_pixels = tile_size.width() * tile_size.height();
    let wrapped_height = (total_pixels as f64 / problem.output_width as f64).ceil() as usize;

    // Clamp the effective tile dimensions to the actual output boundaries.
    let effective_width = tile_size.width().min(problem.output_width);
    let effective_height = wrapped_height
        .max(tile_size.height())
        .min(problem.output_height);

    // Calculate the maximum distance this tile covers in the input image.
    let span_width = ((effective_width as f64 - 1.0) * scale_width).max(0.0);
    let span_height = ((effective_height as f64 - 1.0) * scale_height).max(0.0);

    // Add halo required by the specific interpolation mode.
    let halo = get_halo(options.mode);
    let smem_width = span_width.ceil() as usize + halo + 1;
    let smem_height = span_height.ceil() as usize + halo + 1;

    (smem_width.max(1), smem_height.max(1))
}
