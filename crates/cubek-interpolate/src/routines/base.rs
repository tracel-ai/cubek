use crate::{
    InterpolateError,
    definition::{InterpolateForwardProblem, InterpolateOptions, TileSize, is_flattened},
    routines::InterpolateBlueprint,
};
use cubecl::prelude::*;

#[derive(Debug, Clone)]
pub struct InterpolateLaunchSettings {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub tile_size: TileSize,
    pub num_tiles_width: usize,
    pub num_tiles_height: usize,
    pub num_vectors: usize,
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

pub fn compute_layout<R: Runtime>(
    client: &ComputeClient<R>,
    working_units: usize,
    num_vectors: usize,
    options: InterpolateOptions,
) -> (CubeDim, TileSize) {
    let cube_dim = CubeDim::new(client, working_units);
    let tile_size = TileSize::new(
        cube_dim.y as usize,
        cube_dim.x as usize / num_vectors, // Adjust tile width based on the number of vector
        options,
    );
    (cube_dim, tile_size)
}

pub fn build_settings(
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    cube_dim: CubeDim,
    tile_size: TileSize,
    num_vectors: usize,
) -> InterpolateLaunchSettings {
    let (num_tiles_width, num_tiles_height) = compute_number_of_tiles(problem, tile_size, options);

    let cube_count = CubeCount::Static(
        num_tiles_width as u32,
        num_tiles_height as u32,
        problem.batch as u32,
    );

    InterpolateLaunchSettings {
        cube_count,
        cube_dim,
        tile_size,
        num_tiles_width,
        num_tiles_height,
        num_vectors,
    }
}

pub fn compute_number_of_tiles(
    problem: &InterpolateForwardProblem,
    tile_size: TileSize,
    options: InterpolateOptions,
) -> (usize, usize) {
    if is_flattened(options) {
        // Calculate the number of tiles needed to cover the output, and dispatch in a 1D grid.
        const MAX_DISPATCH: usize = 65535;
        let num_tiles = (problem.output_width * problem.output_height).div_ceil(tile_size.width());
        (
            num_tiles.min(MAX_DISPATCH),
            num_tiles.div_ceil(MAX_DISPATCH),
        )
    } else {
        (
            problem.output_width.div_ceil(tile_size.width()),
            problem.output_height.div_ceil(tile_size.height()),
        )
    }
}
