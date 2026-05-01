use cubecl::{
    prelude::*,
    {self},
};

use crate::tile::ops::{LOGIT_MASKED, Mask, MaskExpand, RowWise};
use crate::tile::scope::Scope;
use crate::tile::{StridedTile, Tile};

#[derive(CubeType)]
pub struct UnitTile<E: Numeric> {
    pub data: Array<E>,
    #[cube(comptime)]
    pub layout: UnitTileLayout,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
// Assumes row-major. If loading from a col-major source, use transposed_load=true
pub struct UnitTileLayout {
    pub num_rows: u32,
    pub num_cols: u32,
    pub transposed_load: bool,
}

impl UnitTileLayout {
    pub const fn new(num_rows: u32, num_cols: u32, transposed_load: bool) -> UnitTileLayout {
        UnitTileLayout {
            num_rows,
            num_cols,
            transposed_load,
        }
    }
}

#[cube]
impl<E: Numeric> UnitTile<E> {
    pub fn new(#[comptime] layout: UnitTileLayout) -> UnitTile<E> {
        let data = Array::<E>::new(comptime!(layout.num_rows * layout.num_cols) as usize);
        UnitTile::<E> { data, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.num_rows * self.layout.num_cols {
            self.data[i as usize] = E::from_int(0);
        }
    }

    pub fn get(&self, row: u32, col: u32) -> E {
        self.data[(row * self.layout.num_cols + col) as usize]
    }

    pub fn accumulate(&mut self, row: u32, col: u32, val: E) {
        self.data[(row * self.layout.num_cols + col) as usize] += val;
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] = self.data[index as usize] * scale.vals[r];
            }
        }
    }

    pub fn rowwise_max(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.num_rows) as usize;
        let num_cols = comptime!(self.layout.num_cols) as usize;
        let mut vals = Array::new(num_rows);

        for r in 0..num_rows {
            let row_offset = r * num_cols;
            let mut val = E::min_value();

            for c in 0..num_cols {
                let index = row_offset + c;
                val = max(val, self.data[index]);
            }

            vals[r] = val;
        }

        RowWise::<E> { num_rows, vals }
    }

    pub fn rowwise_sum(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.num_rows) as usize;
        let num_cols = comptime!(self.layout.num_cols) as usize;
        let mut vals = Array::new(num_rows);

        for r in 0..num_rows {
            let row_offset = r * num_cols;
            let mut val = E::from_int(0);

            for c in 0..num_cols {
                let index = row_offset + c;
                val += self.data[index];
            }

            vals[r] = val;
        }

        RowWise::<E> { num_rows, vals }
    }

    // TODO find a way to have this not necessary if E == E2
    // TODO even if E != E2 it could be written as output to UnitTile::exp_diff rather than exp_diff being inplace
    pub fn copy_from<E2: Numeric>(&mut self, other: &UnitTile<E2>) {
        // Assume layouts are the same

        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] = E::cast_from(other.data[index as usize]);
            }
        }
    }

    pub fn load_from_strided_tile<E2: Numeric, ES: Size>(&mut self, tile: &StridedTile<E2, ES>) {
        if comptime!(self.layout.transposed_load) {
            strided_tile_to_transposed_unit_tile(tile, self)
        } else {
            strided_tile_to_unit_tile(tile, self)
        }
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] = self.data[index as usize] * scale
                    + E::cast_from(mask.should_mask((r, c))) * E::min_value();
            }
        }
    }
}

#[cube]
impl<E: Float> UnitTile<E> {
    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        let num_rows = comptime!(self.layout.num_rows) as usize;
        let num_cols = comptime!(self.layout.num_cols) as usize;
        let threshold = E::new(LOGIT_MASKED);

        for r in 0..num_rows {
            let row_offset = r * num_cols;

            let val = rowwise.vals[r];

            for c in 0..num_cols {
                let index = row_offset + c;

                let safe_val = clamp_min(val, threshold);
                let not_masked = E::cast_from(val >= threshold);
                self.data[index] = not_masked * (self.data[index] - safe_val).exp();
            }
        }
    }
}

#[cube]
/// Allocates a `Tile::Unit`. The variant is valid in any scope — each unit
/// just holds its own row-major copy of the tile.
pub fn allocate_unit_tile<E: Numeric, Sc: Scope>(
    #[comptime] layout: UnitTileLayout,
) -> Tile<E, Sc, ReadWrite> {
    Tile::new_Unit(UnitTile::<E>::new(layout))
}

#[cube]
fn strided_tile_to_unit_tile<E: Numeric, N: Size, E2: Numeric>(
    strided_tile: &StridedTile<E, N>,
    unit_tile: &mut UnitTile<E2>,
) {
    let vector_size = N::value().comptime() as u32;
    assert!(unit_tile.layout.num_cols.is_multiple_of(vector_size));

    let col_iterations = comptime!(unit_tile.layout.num_cols / vector_size);

    for row in 0..unit_tile.layout.num_rows {
        for col in 0..col_iterations {
            let line_read = strided_tile.get_vector(row, col);
            #[unroll]
            for i in 0..vector_size {
                unit_tile.data
                    [(row * unit_tile.layout.num_cols + col * vector_size + i) as usize] =
                    E2::cast_from(line_read[i as usize]);
            }
        }
    }
}

#[cube]
fn strided_tile_to_transposed_unit_tile<E: Numeric, N: Size, E2: Numeric>(
    strided_tile: &StridedTile<E, N>,
    unit_tile: &mut UnitTile<E2>,
) {
    let vector_size = N::value().comptime() as u32;
    assert!(unit_tile.layout.num_cols.is_multiple_of(vector_size));

    let input_num_rows = unit_tile.layout.num_cols.comptime();
    let input_num_cols = unit_tile.layout.num_rows.comptime();
    let vector_iterations = input_num_cols / vector_size;

    for input_row in 0..input_num_rows {
        for input_col_vector in 0..vector_iterations {
            let vector_read = strided_tile.get_vector(input_row, input_col_vector);

            #[unroll]
            for i in 0..vector_size {
                unit_tile.data[((input_col_vector + i) * input_num_rows + input_row) as usize] =
                    E2::cast_from(vector_read[i as usize]);
            }
        }
    }
}
