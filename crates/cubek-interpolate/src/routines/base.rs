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

pub fn build_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    cube_dim: CubeDim,
    tile_size: TileSize,
    num_vectors: usize,
) -> InterpolateLaunchSettings {
    let (num_tiles_width, num_tiles_height) = compute_number_of_tiles(problem, tile_size, options);

    let cube_count = compute_cube_count(client, problem, num_tiles_width, num_tiles_height);

    InterpolateLaunchSettings {
        cube_count,
        cube_dim,
        tile_size,
        num_tiles_width,
        num_tiles_height,
        num_vectors,
    }
}

fn compute_number_of_tiles(
    problem: &InterpolateForwardProblem,
    tile_size: TileSize,
    options: InterpolateOptions,
) -> (usize, usize) {
    if is_flattened(options) {
        let num_tiles = (problem.output_width * problem.output_height).div_ceil(tile_size.width());
        // All tiles are arranged in a single row
        (num_tiles, 1)
    } else {
        (
            problem.output_width.div_ceil(tile_size.width()),
            problem.output_height.div_ceil(tile_size.height()),
        )
    }
}

fn compute_cube_count<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    num_tiles_width: usize,
    num_tiles_height: usize,
) -> CubeCount {
    let (max_cube_count_x, max_cube_count_y, max_cube_count_z) =
        client.properties().hardware.max_cube_count;

    let total_cube_count = (num_tiles_width * num_tiles_height * problem.batch) as u32;

    let cube_count_x = total_cube_count.min(max_cube_count_x);

    let required_cube_count_y = total_cube_count.div_ceil(cube_count_x);
    let cube_count_y = required_cube_count_y.min(max_cube_count_y);

    let cube_count_z = required_cube_count_y.div_ceil(cube_count_y);

    assert!(
        cube_count_z <= max_cube_count_z,
        "Total work volume ({}) exceeds maximum 3D dispatch limits of the GPU.",
        total_cube_count
    );

    CubeCount::Static(cube_count_x, cube_count_y, cube_count_z)
}
