use crate::{
    InterpolateError,
    definition::{
        InterpolateForwardProblem, InterpolateOptions, InterpolateProblem, TileSize, get_halo,
    },
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
pub enum BlueprintStrategy<R: Routine> {
    Forced(R::Blueprint),
    Inferred(R::Strategy),
}

pub trait Routine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: InterpolateProblem,
        strategy: BlueprintStrategy<Self>,
        bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError>;
}

pub(crate) fn prepare_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateProblem,
    options: InterpolateOptions,
    bytes_per_element: usize,
    vector_size: usize,
    max_shared_memory_bytes: Option<usize>,
) -> Result<InterpolateLaunchSettings, InterpolateError> {
    let InterpolateProblem::Forward(InterpolateForwardProblem {
        input_shape,
        output_size,
        ..
    }) = problem
    else {
        return Err(InterpolateError::UnsupportedMode(
            "backward interpolation can't be performed with forward kernel".to_string(),
        ));
    };

    let batch = input_shape[0];
    let (output_width, output_height) = (output_size[0], output_size[1]);
    let (input_width, input_height) = (input_shape[2], input_shape[1]);
    let channel_groups = input_shape[3] / vector_size;

    let mut working_units = output_width * output_height * batch * channel_groups;

    let (cube_dim, tile_size, smem_width, smem_height) = loop {
        let cube_dim = CubeDim::new(client, working_units);
        let tile_size = TileSize::new(cube_dim.x as usize, cube_dim.y as usize, options);

        let (smem_width, smem_height) = match max_shared_memory_bytes {
            Some(max_shared_memory_bytes) => {
                let (smem_width, smem_height) = compute_smem_size(
                    input_width,
                    input_height,
                    output_width,
                    output_height,
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

    let num_tiles_x = output_width.div_ceil(tile_size.width());
    let num_tiles_y = output_height.div_ceil(tile_size.height());

    let cube_count = CubeCount::Static(
        (num_tiles_x * channel_groups) as u32,
        num_tiles_y as u32,
        batch as u32,
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

    let (tile_w, tile_h) = if output_tile_size.is_row_vector() {
        (
            output_width.max(1) as f64,
            output_tile_size.area().div_ceil(output_width).max(1) as f64,
        )
    } else {
        (
            output_tile_size.width().max(1) as f64,
            output_tile_size.height().max(1) as f64,
        )
    };

    let scale_x = input_width as f64 / output_width as f64;
    let scale_y = input_height as f64 / output_height as f64;

    // Calculate the distance between the first and last pixel.
    let span_x = ((tile_w - 1.0) * scale_x).max(0.0);
    let span_y = ((tile_h - 1.0) * scale_y).max(0.0);

    // Halo is added half on each side.
    let smem_w = span_x.ceil() as usize + halo;
    let smem_h = span_y.ceil() as usize + halo;

    (smem_w.max(1), smem_h.max(1))
}
