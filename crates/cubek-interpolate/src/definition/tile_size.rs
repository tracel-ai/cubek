use crate::definition::{InterpolateOptions, get_halo, is_flattened};
use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    height: usize,
    width: usize,
}

impl TileSize {
    pub fn new(height: usize, width: usize, options: InterpolateOptions) -> Self {
        let halo = get_halo(options.mode);
        let area = width * height;

        if !is_flattened(options) && area.is_multiple_of(halo) {
            Self {
                height: halo,
                width: area / halo,
            }
        } else {
            Self {
                height: 1,
                width: area,
            }
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

#[cube]
pub fn tile_absolute_coords(
    output_width: usize,
    cube_pos: usize,
    unit_pos: usize,
    #[comptime] tile_size: TileSize,
    #[comptime] options: InterpolateOptions,
) -> (usize, usize) {
    if is_flattened(options) {
        let flat = cube_pos * tile_size.width() + unit_pos;
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
