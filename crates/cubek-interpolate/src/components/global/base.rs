use crate::components::global::TileSize;
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn interpolate_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    #[comptime] output_tile_size: TileSize,
    #[define(F)] _dtype: StorageType,
) {
    let (rem, channel_group) = cube_shape[0].div_mod(ABSOLUTE_POS);
    let (rem, unit_pos) = cube_shape[1].div_mod(rem);
    let (batch, cube_pos) = cube_shape[2].div_mod(rem);

    let (output_width, output_height) = (output.shape(2), output.shape(1));

    let (x, y) = compute_absolute_coords(
        output_tile_size,
        output_width,
        output_height,
        cube_pos,
        unit_pos,
    );

    if x >= output_width || y >= output_height {
        terminate!();
    }

    let (input_width, input_height) = (input.shape(2), input.shape(1));

    let in_y = (y * input_height) / output_height;
    let in_x = (x * input_width) / output_width;

    let vector_size = input.vector_size();

    let in_nearest_index =
        (batch * input.stride(0) + in_y * input.stride(1) + in_x * input.stride(2)) / vector_size
            + channel_group * input.stride(3);

    let out_index = (batch * output.stride(0) + y * output.stride(1) + x * output.stride(2))
        / vector_size
        + channel_group * output.stride(3);

    output[out_index] = input[in_nearest_index];
}

#[cube]
fn compute_absolute_coords(
    #[comptime] tile: TileSize,
    output_width: usize,
    _output_height: usize,
    cube_pos: usize,
    unit_pos: usize,
) -> (usize, usize) {
    if tile.is_row_vector() {
        let flat = cube_pos * tile.width() + unit_pos;
        (flat % output_width, flat / output_width)
    } else {
        let num_tiles_x = output_width.div_ceil(tile.width());

        let (local_x, local_y) = local_coords(tile, unit_pos);
        let (cube_x, cube_y) = cube_coords(cube_pos, num_tiles_x);

        (
            cube_x * tile.width() + local_x,
            cube_y * tile.height() + local_y,
        )
    }
}

#[cube]
fn local_coords(#[comptime] tile: TileSize, unit_pos: usize) -> (usize, usize) {
    if tile.is_row_vector() {
        (unit_pos, 0)
    } else {
        (unit_pos % tile.width(), unit_pos / tile.width())
    }
}

#[cube]
fn cube_coords(cube_pos: usize, num_tiles_x: usize) -> (usize, usize) {
    (cube_pos % num_tiles_x, cube_pos / num_tiles_x)
}
