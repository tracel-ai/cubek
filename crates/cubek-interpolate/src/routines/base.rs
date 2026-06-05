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
    pub cubes_per_batch: usize,
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
        vector_size: usize,
        bytes_per_element: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError>;
}

pub fn compute_layout<R: Runtime>(
    client: &ComputeClient<R>,
    working_units: usize,
    tile_target_aspect_ratio: f32,
    options: InterpolateOptions,
) -> (CubeDim, TileSize, usize) {
    let cube_dim = CubeDim::new(client, working_units);

    let units_per_cube = cube_dim.x as usize * cube_dim.y as usize * cube_dim.z as usize;

    // HARDCODED for testing
    let tile_area = 256;

    let tile_size = TileSize::new(tile_area, tile_target_aspect_ratio, options);

    (cube_dim, tile_size, units_per_cube)
}

pub fn build_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    options: InterpolateOptions,
    cube_dim: CubeDim,
    tile_size: TileSize,
    num_vectors: usize,
) -> InterpolateLaunchSettings {
    let cubes_per_batch = compute_cubes_per_batch(problem, tile_size, options);

    let cube_count = compute_cube_count(client, problem, cubes_per_batch);

    InterpolateLaunchSettings {
        cube_count,
        cube_dim,
        tile_size,
        cubes_per_batch,
        num_vectors,
    }
}

fn compute_cubes_per_batch(
    problem: &InterpolateForwardProblem,
    tile_size: TileSize,
    options: InterpolateOptions,
) -> usize {
    if is_flattened(options) {
        let total_pixels = problem.output_width * problem.output_height;

        total_pixels.div_ceil(tile_size.width())
    } else {
        let num_tiles_width = problem.output_width.div_ceil(tile_size.width());
        let num_tiles_height = problem.output_height.div_ceil(tile_size.height());

        num_tiles_width * num_tiles_height
    }
}

fn compute_cube_count<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    cubes_per_batch: usize,
) -> CubeCount {
    let (max_cube_count_x, max_cube_count_y, max_cube_count_z) =
        client.properties().hardware.max_cube_count;

    let total_cube_count = (cubes_per_batch * problem.batch) as u32;

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
