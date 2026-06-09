use crate::{
    InterpolateError,
    definition::{InterpolateForwardProblem, TileSize},
    routines::InterpolateBlueprint,
};
use cubecl::prelude::*;

#[derive(Debug, Clone)]
pub struct InterpolateLaunchSettings {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
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

pub fn build_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &InterpolateForwardProblem,
    cube_dim: CubeDim,
    tile_size: TileSize,
    is_flattened: bool,
    num_vectors: usize,
) -> InterpolateLaunchSettings {
    let cubes_per_batch = compute_cubes_per_batch(problem, tile_size, is_flattened);

    let cube_count = compute_cube_count(client, problem, cubes_per_batch);

    InterpolateLaunchSettings {
        cube_count,
        cube_dim,
        cubes_per_batch,
        num_vectors,
    }
}

fn compute_cubes_per_batch(
    problem: &InterpolateForwardProblem,
    tile_size: TileSize,
    is_flattened: bool,
) -> usize {
    if is_flattened {
        let total_pixels = problem.output_width * problem.output_height;

        total_pixels.div_ceil(tile_size.area())
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
