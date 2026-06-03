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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryStrategy {
    pub shared_memory_height: usize,
}

impl ForwardRoutine for SharedMemoryRoutine {
    type Strategy = SharedMemoryStrategy;
    type Blueprint = SharedMemoryBlueprint;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        _strategy: BlueprintStrategy<Self>,
        bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let (settings, smem_width, smem_height) = prepare_shared_launch_settings(
            client,
            problem,
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
    bytes_per_element: usize,
    vector_size: usize,
    max_shared_memory_bytes: usize,
) -> Result<(InterpolateLaunchSettings, usize, usize), InterpolateError> {
    let num_vectors = problem.channels / vector_size;
    let mut working_units = problem.output_width * problem.output_height * num_vectors;

    loop {
        let (cube_dim, tile_size) =
            compute_layout(client, working_units, num_vectors, problem.options);
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
            // Stop looping when 1 working unit cannot be further divided.
            if working_units <= 1 {
                return Err(InterpolateError::SharedMemoryLimitExceeded {
                    requested: requested_smem_bytes,
                    available: max_shared_memory_bytes,
                });
            }

            working_units = (working_units / 2).max(1);
        }
    }
}

fn compute_smem_size(
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    output_tile_size: TileSize,
) -> (usize, usize) {
    let scale_height = problem.input_height as f64 / problem.output_height as f64;
    let scale_width = problem.input_width as f64 / problem.output_width as f64;

    // Calculate the distance between the first and last pixel.
    let span_height = ((output_tile_size.height() as f64 - 1.0) * scale_height).max(0.0);
    let span_width = ((output_tile_size.width() as f64 - 1.0) * scale_width).max(0.0);

    // Halo is added half on each side.
    let halo = get_halo(options.mode);
    let smem_height = span_height.ceil() as usize + halo + 1;
    let smem_width = span_width.ceil() as usize + halo + 1;

    (smem_width.max(1), smem_height.max(1))
}
