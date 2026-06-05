use crate::routines::InterpolateBlueprint;
use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    height: usize,
    width: usize,
}

impl TileSize {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    pub fn area(&self) -> usize {
        self.width * self.height
    }
}

#[cube]
pub fn tile_absolute_coords(
    output_width: usize,
    cube_pos: usize,
    unit_pos: usize,
    #[comptime] blueprint: InterpolateBlueprint,
) -> (usize, usize) {
    let tile_size = blueprint.tile_size;

    if blueprint.is_flattened() {
        let flat = cube_pos * tile_size.area() + unit_pos;
        (flat / output_width, flat % output_width)
    } else {
        let num_col = output_width.div_ceil(tile_size.width());

        let (local_row, local_col) = tile_local_coords(unit_pos, tile_size);
        let (cube_row, cube_col) = tile_cube_coords(cube_pos, num_col);

        (
            cube_row * tile_size.height() + local_row,
            cube_col * tile_size.width() + local_col,
        )
    }
}

#[cube]
fn tile_local_coords(unit_pos: usize, #[comptime] tile_size: TileSize) -> (usize, usize) {
    (unit_pos / tile_size.width(), unit_pos % tile_size.width())
}

#[cube]
fn tile_cube_coords(cube_pos: usize, num_col: usize) -> (usize, usize) {
    (cube_pos / num_col, cube_pos % num_col)
}
