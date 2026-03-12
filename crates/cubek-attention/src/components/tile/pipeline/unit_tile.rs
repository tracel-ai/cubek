use cubecl;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;

use crate::components::tile::{
    LOGIT_MASKED,
    pipeline::{RowVal, RowWise},
    softmax::{FragmentMask, FragmentMaskExpand, SoftmaxLayout, SoftmaxLayoutExpand},
};

#[derive(CubeType)]
pub struct UnitTile<E: Numeric> {
    pub data: Array<E>,
    pub layout: UnitTileLayout,
}

#[derive(CubeType, Copy, Clone)]
pub struct UnitTileLayout {
    #[cube(comptime)]
    pub num_rows: u32,
    #[cube(comptime)]
    pub num_cols: u32,
}

#[cube]
impl<E: Numeric> UnitTile<E> {
    pub fn new(layout: UnitTileLayout) -> UnitTile<E> {
        let data = Array::<E>::new(comptime!(layout.num_rows * layout.num_cols) as usize);
        UnitTile::<E> { data, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.num_rows * self.layout.num_cols {
            self.data[i as usize] = E::from_int(0);
        }
    }

    pub fn get(&self, i: u32, j: u32) -> E {
        self.data[(i * self.layout.num_cols + j) as usize]
    }

    pub fn accumulate(&mut self, i: u32, j: u32, val: E) {
        self.data[(i * self.layout.num_cols + j) as usize] += val;
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;
            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] = self.data[index as usize] * scale.index(r);
            }
        }
    }

    pub fn rowwise_max(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            let mut val = E::min_value();

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                val = max(val, self.data[index as usize]);
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.num_rows.comptime() as usize,
            vals,
        }
    }

    pub fn rowwise_sum(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            let mut val = E::from_int(0);

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                val += self.data[index as usize];
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.num_rows.comptime() as usize,
            vals,
        }
    }

    pub fn scale_and_mask<M: FragmentMask>(&mut self, scale: E, mask: &M) {
        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] = self.data[index as usize] * scale
                    + E::cast_from(mask.should_mask((r, c).runtime())) * E::min_value();
            }
        }
    }

    pub fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
    }
}

#[cube]
impl<E: Float> UnitTile<E> {
    pub fn exp_diff(&mut self, val: &RowWise<E>) {
        let threshold = E::new(LOGIT_MASKED);

        #[unroll]
        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;

            let val = val.index(r);

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;

                let safe_val = clamp_min(val, threshold);
                let not_masked = E::cast_from(val >= threshold);
                self.data[index as usize] =
                    not_masked * (self.data[index as usize] - safe_val).exp();
            }
        }
    }
}

#[cube]
impl UnitTileLayout {
    pub fn new(#[comptime] num_rows: u32, #[comptime] num_cols: u32) -> UnitTileLayout {
        UnitTileLayout { num_rows, num_cols }
    }
}

#[cube]
impl SoftmaxLayout for UnitTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        local_pos
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        1u32
    }
}

#[cube]
impl<E: Numeric> FragmentMask for UnitTile<E> {
    type Layout = UnitTileLayout;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.data[(local_pos.0 * self.layout.num_cols + local_pos.1) as usize])
    }
}
