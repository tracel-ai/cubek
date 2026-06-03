use crate::definition::{InterpolateOptions, is_flattened};
use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    height: usize,
    width: usize,
}

impl TileSize {
    /// Creates a tile shape with the given area, choosing dimensions whose
    /// width-to-height ratio is as close as possible to the target aspect ratio.
    ///
    /// The returned tile always satisfies:
    ///
    /// ```text
    /// width * height == area
    /// ```
    ///
    /// When multiple shapes are equally good matches, wider layouts are preferred
    /// over taller ones. If the target aspect ratio is invalid, the area is zero,
    /// or the operation uses a flattened layout, a 1D tile `(1, area)` is returned.
    pub fn new(area: usize, tile_target_aspect_ratio: f32, options: InterpolateOptions) -> Self {
        if tile_target_aspect_ratio <= 0.0 || area == 0 || is_flattened(options) {
            return Self {
                height: 1,
                width: area,
            };
        }

        let score = |h: usize| {
            let w = area / h;
            let ratio = w as f32 / h as f32;

            let error = ratio.max(tile_target_aspect_ratio) / ratio.min(tile_target_aspect_ratio);

            // Prefer wider layouts when the error is identical.
            if ratio < tile_target_aspect_ratio {
                error * 1.01
            } else {
                error
            }
        };

        let limit = (area as f64).sqrt() as usize;

        let best_height = (1..=limit)
            .filter(|&h| area.is_multiple_of(h))
            .flat_map(|h| [h, area / h])
            .min_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap())
            .unwrap_or(1);

        Self {
            height: best_height,
            width: area / best_height,
        }
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
