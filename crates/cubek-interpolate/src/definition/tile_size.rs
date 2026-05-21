use crate::definition::{InterpolateOptions, get_halo};
use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    w: usize,
    h: usize,
}

impl TileSize {
    pub fn new(w: usize, h: usize, options: InterpolateOptions) -> Self {
        if get_halo(options.mode) == 1 {
            return Self { w: w * h, h: 1 };
        }
        Self { w, h }
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn area(&self) -> usize {
        self.w * self.h
    }

    pub fn is_row_vector(&self) -> bool {
        self.h == 1
    }
}

#[cube]
pub fn tile_absolute_coords(
    output_width: usize,
    cube_pos: usize,
    unit_pos: usize,
    #[comptime] output_tile_size: TileSize,
) -> (usize, usize) {
    if output_tile_size.is_row_vector() {
        let flat = cube_pos * output_tile_size.width() + unit_pos;
        (flat % output_width, flat / output_width)
    } else {
        let num_tiles_x = output_width.div_ceil(output_tile_size.width());

        let (local_x, local_y) = tile_local_coords(unit_pos, output_tile_size);
        let (cube_x, cube_y) = tile_cube_coords(cube_pos, num_tiles_x);

        (
            cube_x * output_tile_size.width() + local_x,
            cube_y * output_tile_size.height() + local_y,
        )
    }
}

#[cube]
fn tile_local_coords(unit_pos: usize, #[comptime] output_tile_size: TileSize) -> (usize, usize) {
    if output_tile_size.is_row_vector() {
        (unit_pos, 0)
    } else {
        (
            unit_pos % output_tile_size.width(),
            unit_pos / output_tile_size.width(),
        )
    }
}

#[cube]
fn tile_cube_coords(cube_pos: usize, num_tiles_x: usize) -> (usize, usize) {
    (cube_pos % num_tiles_x, cube_pos / num_tiles_x)
}
