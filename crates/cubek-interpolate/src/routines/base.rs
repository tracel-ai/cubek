use crate::{
    InterpolateError,
    definition::{InterpolateForwardProblem, InterpolateOptions, TileSize, get_halo},
    routines::InterpolateBlueprint,
};
use cubecl::prelude::*;

#[derive(Debug, Clone)]
pub struct InterpolateLaunchSettings {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub tile_size: TileSize,
    pub smem_width: usize,
    pub smem_height: usize,
    pub channel_groups: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlueprintStrategy<R: ForwardRoutine> {
    Forced(R::Blueprint),
    Inferred(R::Strategy),
}

pub trait ForwardRoutine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        strategy: BlueprintStrategy<Self>,
        bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError>;
}

pub(crate) fn prepare_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    bytes_per_element: usize,
    vector_size: usize,
    max_shared_memory_bytes: Option<usize>,
) -> Result<InterpolateLaunchSettings, InterpolateError> {
    let channel_groups = problem.channels / vector_size;

    let mut working_units =
        problem.output_width * problem.output_height * problem.batch * channel_groups;

    let (cube_dim, tile_size, smem_width, smem_height) = loop {
        let cube_dim = CubeDim::new(client, working_units);

        let tile_size = TileSize::new(cube_dim.x as usize, cube_dim.y as usize, options);

        let (smem_width, smem_height) = match max_shared_memory_bytes {
            Some(max_shared_memory_bytes) => {
                let (smem_width, smem_height) = compute_smem_size(
                    problem.input_width,
                    problem.input_height,
                    problem.output_width,
                    problem.output_height,
                    options,
                    tile_size,
                );

                let requested_smem_bytes = smem_width * smem_height * bytes_per_element;

                if requested_smem_bytes <= max_shared_memory_bytes {
                    break (cube_dim, tile_size, smem_width, smem_height);
                }

                if working_units <= 1 {
                    return Err(InterpolateError::SharedMemoryLimitExceeded {
                        requested: requested_smem_bytes,
                        available: max_shared_memory_bytes,
                    });
                }

                working_units = (working_units / 2).max(1);
                continue;
            }
            None => (0, 0),
        };

        break (cube_dim, tile_size, smem_width, smem_height);
    };

    let (num_tiles_width, num_tiles_height) = (
        problem.output_width.div_ceil(tile_size.width()),
        problem.output_height.div_ceil(tile_size.height()),
    );

    let cube_count = CubeCount::Static(
        (num_tiles_width * channel_groups) as u32,
        num_tiles_height as u32,
        problem.batch as u32,
    );

    Ok(InterpolateLaunchSettings {
        cube_count,
        cube_dim,
        tile_size,
        smem_width,
        smem_height,
        channel_groups,
    })
}

fn compute_smem_size(
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    options: InterpolateOptions,
    output_tile_size: TileSize,
) -> (usize, usize) {
    let halo = get_halo(options.mode);

    let scale_x = input_width as f64 / output_width as f64;
    let scale_y = input_height as f64 / output_height as f64;

    // Calculate the distance between the first and last pixel.
    let span_x = ((output_tile_size.width() as f64 - 1.0) * scale_x).max(0.0);
    let span_y = ((output_tile_size.height() as f64 - 1.0) * scale_y).max(0.0);

    // Halo is added half on each side.
    let smem_w = span_x.ceil() as usize + halo;
    let smem_h = span_y.ceil() as usize + halo;

    (smem_w.max(1), smem_h.max(1))
}
