use cubecl::std::tensor::layout::Coords2d;
use cubecl::{self, prelude::*};

use crate::tile::{
    LOGIT_MASKED, Plane, RowWise, StridedTile, Tile, TileKind, TileKindExpand,
    mask::{Mask, MaskExpand},
    scope::TileScope,
};

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

    /// Reads the element at `local_pos` and casts to `bool`. Used by the
    /// `Mask` trait dispatcher when this tile is acting as a materialized
    /// mask fragment.
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.data[(local_pos.0 * self.layout.num_cols + local_pos.1) as usize])
    }

    pub fn accumulate(&mut self, row: u32, col: u32, val: E) {
        self.data[(row * self.layout.num_cols + col) as usize] += val;
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] *= scale.vals[r];
            }
        }
    }

    pub fn rowwise_max(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.num_rows) as usize;
        let num_cols = comptime!(self.layout.num_cols) as usize;
        let mut vals = Array::<E>::new(num_rows);

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
        let mut vals = Array::<E>::new(num_rows);

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
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        acc.copy_from(base);
        acc.max_inplace(&self.rowwise_max());
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        acc.fill(E::from_int(0));
        acc.add_inplace(&self.rowwise_sum());
    }

    pub fn fill_zero(&mut self) {
        self.zero();
    }

    /// Cast-copies this unit tile into `dest`. Used by per-variant softmax
    /// helpers when writing the post-softmax score into a same-storage
    /// destination.
    pub fn write_to<Lhs: Float>(&self, dest: &mut UnitTile<Lhs>) {
        let total = comptime!(self.layout.num_rows * self.layout.num_cols);
        for i in 0..total {
            dest.data[i as usize] = Lhs::cast_from(self.data[i as usize]);
        }
    }

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
pub fn allocate_unit_tile<E: Numeric, Sc: TileScope>(
    #[comptime] layout: UnitTileLayout,
) -> Tile<E, Sc> {
    Tile::from_kind(TileKind::new_Unit(UnitTile::<E>::new(layout)))
}

#[cube]
impl<Acc: Float> UnitTile<Acc> {
    /// Online softmax for the per-unit-full Unit variant. Each unit holds the
    /// whole tile in registers; reductions are register-local. Destination
    /// must be another `UnitTile`.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed: &mut Tile<Lhs, Plane>,
        state: &mut (RowWise<Acc>, RowWise<Acc>),
        head_dim_factor: Acc,
    ) -> RowWise<Acc> {
        let num_rows = comptime!(state.0.num_rows);
        let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
        let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

        self.scale_and_mask::<M>(head_dim_factor, mask);
        self.row_max(&mut max_buf, &state.0);
        self.exp_diff(&max_buf);
        self.row_sum(&mut sum_buf);

        let exp_m_diff = state.0.exp_diff(&max_buf);
        let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

        match &mut softmaxed.kind {
            TileKind::Unit(d) => self.write_to::<Lhs>(d),
            TileKind::Bounce(_) => panic!("UnitTile::softmax: Bounce destination not supported"),
            TileKind::WhiteboxFragment(_) => {
                panic!("UnitTile::softmax: WhiteboxFragment destination not supported")
            }
            TileKind::Register(_) => {
                panic!("UnitTile::softmax: Register destination not supported")
            }
            _ => panic!("UnitTile::softmax: unsupported softmaxed variant"),
        }

        RowWise::copy_from(&mut state.0, &max_buf);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }
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
                    E2::cast_from(line_read.extract(i as usize));
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
                    E2::cast_from(vector_read.extract(i as usize));
            }
        }
    }
}
