use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::ops::{LOGIT_MASKED, Mask, MaskExpand, RowWise};
use crate::tile::scope::{Scope, assert_plane_scope};
use crate::tile::{StridedTile, Tile};

#[derive(CubeType)]
/// Assumes:
/// - unit_size * plane_dim = total_size (not dim wise but in total count)
pub struct LocalTile<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: LocalTileLayout,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum InnerLayout {
    /// Each unit has all its elements contiguous inside the same row
    ///
    ///  0,  0,  1,  1,  2,  2,  3,  3,
    ///  4,  4,  5,  5,  6,  6,  7,  7,
    ///  8,  8,  9,  9, 10, 10, 11, 11,
    /// 12, 12, 13, 13, 14, 14, 15, 15,
    /// 16, 16, 17, 17, 18, 18, 19, 19,
    /// 20, 20, 21, 21, 22, 22, 23, 23,
    /// 24, 24, 25, 25, 26, 26, 27, 27,
    /// 28, 28, 29, 29, 30, 30, 31, 31,
    Contiguous,
    /// Each unit spreads its elements along two rows
    ///
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    SplitRows,
}

#[cube]
impl<E: Numeric> LocalTile<E> {
    pub fn new(#[comptime] layout: LocalTileLayout) -> LocalTile<E> {
        let array = Array::<E>::new(comptime!(layout.unit_size.0 * layout.unit_size.1) as usize);

        LocalTile::<E> { array, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.unit_size.0 * self.layout.unit_size.1 {
            self.array[i as usize] = E::from_int(0);
        }
    }

    pub fn load_from_slice(&mut self, smem_slice: &Slice<E>) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = local_layout_absolute_pos(self.layout, (r, c));
                let index = row * self.layout.total_size.1 + col;

                self.array[(r * self.layout.unit_size.1 + c) as usize] = smem_slice[index as usize];
            }
        }
    }

    pub fn load_from_strided_tile<E2: Numeric, N: Size>(
        &mut self,
        strided_tile: &StridedTile<E2, N>,
    ) {
        // Assumes vector size == 1
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = local_layout_absolute_pos(self.layout, (r, c));
                self.array[(r * self.layout.unit_size.1 + c) as usize] =
                    E::cast_from(strided_tile.get_vector(row, col))
            }
        }
    }

    pub fn store_to<F: Float>(&self, smem_slice: &mut SliceMut<F>) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = local_layout_absolute_pos(self.layout, (r, c));
                let index = row * self.layout.total_size.1 + col;

                smem_slice[index as usize] =
                    F::cast_from(self.array[(r * self.layout.unit_size.1 + c) as usize]);
            }
        }
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        for r in 0..self.layout.unit_size.0 as usize {
            let row_offset = r as u32 * self.layout.unit_size.1;
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index as usize] = self.array[index as usize] * scale.vals[r];
            }
        }
    }

    pub fn rowwise_max(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.unit_size.0) as usize;
        let num_cols = comptime!(self.layout.unit_size.1) as usize;
        let mut vals = Array::new(num_rows);

        for r in 0..num_rows {
            let row_offset = r * num_cols;
            let mut val = E::min_value();

            for c in 0..num_cols {
                let index = row_offset + c;
                val = max(val, self.array[index]);
            }

            vals[r] = val;
        }

        RowWise::<E> { num_rows, vals }
    }

    pub fn rowwise_sum(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.unit_size.0) as usize;
        let num_cols = comptime!(self.layout.unit_size.1) as usize;
        let mut vals = Array::new(num_rows);

        for r in 0..num_rows {
            let row_offset = r * num_cols;
            let mut val = E::from_int(0);

            for c in 0..num_cols {
                let index = row_offset + c;
                val += self.array[index];
            }

            vals[r] = val;
        }

        RowWise::<E> { num_rows, vals }
    }

    pub fn num_units_per_row(&self) -> comptime_type!(u32) {
        comptime!(self.layout.total_size.1 / self.layout.unit_size.1)
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index as usize] = self.array[index as usize] * scale
                    + E::cast_from(mask.should_mask((r, c))) * E::min_value();
            }
        }
    }
}

#[cube]
impl<E: Float> LocalTile<E> {
    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        let num_rows = comptime!(self.layout.unit_size.0) as usize;
        let num_cols = comptime!(self.layout.unit_size.1) as usize;
        let threshold = E::new(LOGIT_MASKED);

        for r in 0..num_rows {
            let row_offset = r * num_cols;

            let val = rowwise.vals[r];
            let safe_val = clamp_min(val, threshold);
            let not_masked = E::cast_from(val >= threshold);

            for c in 0..num_cols {
                let index = row_offset + c;

                self.array[index] = not_masked * (self.array[index] - safe_val).exp();
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LocalTileLayout {
    pub total_size: Coords2d,
    pub unit_size: Coords2d,
    pub num_units_per_row: u32,
    pub plane_dim: u32,
}

impl LocalTileLayout {
    pub const fn new(
        total_size: Coords2d,
        plane_dim: u32,
        inner_layout: InnerLayout,
    ) -> LocalTileLayout {
        let total_elements = total_size.0 * total_size.1;
        let elements_per_unit = total_elements.div_ceil(plane_dim);

        let (num_rows_per_unit, num_cols_per_unit) = match inner_layout {
            InnerLayout::Contiguous => (1u32, elements_per_unit),
            InnerLayout::SplitRows => (2u32, elements_per_unit / 2u32),
        };
        let unit_size = (num_rows_per_unit, num_cols_per_unit);
        let num_units_per_row = total_size.1 / unit_size.1;

        LocalTileLayout {
            total_size,
            unit_size,
            num_units_per_row,
            plane_dim,
        }
    }

    pub const fn num_units_per_row(&self) -> u32 {
        self.total_size.1 / self.unit_size.1
    }
}

#[cube]
/// Allocates a `Tile::Local` for the given scope. Panics at expansion time
/// unless `Sc = Plane`.
pub fn allocate_local_tile<E: Numeric, Sc: Scope>(
    #[comptime] layout: LocalTileLayout,
) -> Tile<E, Sc, ReadWrite> {
    comptime!(assert_plane_scope(Sc::KIND));
    Tile::new_Local(LocalTile::<E>::new(layout))
}

/// Maps a per-unit `(row, col)` to its absolute position within the tile
/// described by `layout`.
#[cube]
pub fn local_layout_absolute_pos(
    #[comptime] layout: LocalTileLayout,
    local_pos: Coords2d,
) -> Coords2d {
    let abs_row_index = {
        let row_0 = UNIT_POS_X / layout.num_units_per_row;
        let row_jump = comptime!(layout.plane_dim / layout.num_units_per_row);
        local_pos.0 * row_jump + row_0
    };
    let abs_col_index = layout.unit_size.1 * (UNIT_POS_X % layout.num_units_per_row) + local_pos.1;
    (abs_row_index, abs_col_index)
}

/// Zeroes a slice giving responsibility to units following `layout`.
#[cube]
pub fn local_layout_zero_slice<E: Numeric>(
    #[comptime] layout: LocalTileLayout,
    slice: &mut SliceMut<E>,
) {
    for r in 0..layout.unit_size.0 {
        for c in 0..layout.unit_size.1 {
            let (row, col) = local_layout_absolute_pos(layout, (r, c));
            let index = row * layout.total_size.1 + col;

            slice[index as usize] = E::from_int(0);
        }
    }
}
