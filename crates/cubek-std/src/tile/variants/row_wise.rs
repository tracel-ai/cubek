use cubecl;
use cubecl::prelude::*;

/// Below this value a row is treated as numerically zero. Above f16's
/// smallest normal (~6.1e-5).
pub const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;

#[derive(CubeType)]
/// Contains one value per row of a fragment for which the unit contributes
///
/// Example: For a 8x8 tile shared by a plane of 32 units,
/// every unit holds 8 values in the tile.
///
/// In the following layout, values are held contiguously, and num_rows=1 because
/// every two occurrences of the same plane id are in the same row
///  0,  0,  1,  1,  2,  2,  3,  3,
///  4,  4,  5,  5,  6,  6,  7,  7,
///  8,  8,  9,  9, 10, 10, 11, 11,
/// 12, 12, 13, 13, 14, 14, 15, 15,
/// 16, 16, 17, 17, 18, 18, 19, 19,
/// 20, 20, 21, 21, 22, 22, 23, 23,
/// 24, 24, 25, 25, 26, 26, 27, 27,
/// 28, 28, 29, 29, 30, 30, 31, 31,
///
/// In the following layout, values are held disjointly, and num_rows=2 because
/// the two occurrences of the same plane id are not in the same row
///  0,  1,  2,  3,  4,  5,  6,  7,
///  8,  9, 10, 11, 12, 13, 14, 15,
/// 16, 17, 18, 19, 20, 21, 22, 23,
/// 24, 25, 26, 27, 28, 29, 30, 31,
///  0,  1,  2,  3,  4,  5,  6,  7,
///  8,  9, 10, 11, 12, 13, 14, 15,
/// 16, 17, 18, 19, 20, 21, 22, 23,
/// 24, 25, 26, 27, 28, 29, 30, 31,
pub struct RowWise<E: Numeric> {
    pub vals: Array<E>,
    #[cube(comptime)]
    pub num_rows: usize,
}

#[cube]
impl<E: Numeric> RowWise<E> {
    pub fn new_filled(#[comptime] num_rows: usize, val: E) -> RowWise<E> {
        let mut vals = Array::<E>::new(num_rows);
        for i in 0..num_rows {
            vals[i] = val;
        }
        RowWise::<E> { vals, num_rows }
    }

    pub fn fill(&mut self, val: E) {
        for i in 0..self.num_rows {
            self.vals[i] = val;
        }
    }

    pub fn init_zero(&mut self) {
        self.fill(E::from_int(0));
    }

    pub fn new_min_value(#[comptime] num_rows: usize) -> RowWise<E> {
        Self::new_filled(num_rows, E::min_value())
    }

    pub fn new_zero(#[comptime] num_rows: usize) -> RowWise<E> {
        Self::new_filled(num_rows, E::from_int(0))
    }

    pub fn copy_from(&mut self, other: &RowWise<E>) {
        for i in 0..self.num_rows {
            self.vals[i] = other.vals[i]
        }
    }

    pub fn add(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut result = Array::<E>::new(self.num_rows);
        for i in 0..self.num_rows {
            result[i] = self.vals[i] + other.vals[i];
        }
        RowWise::<E> {
            vals: result,
            num_rows: self.num_rows,
        }
    }

    pub fn add_inplace(&mut self, other: &RowWise<E>) {
        for i in 0..self.num_rows {
            self.vals[i] += other.vals[i];
        }
    }

    pub fn mul(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut result = Array::<E>::new(self.num_rows);
        for i in 0..self.num_rows {
            result[i] = self.vals[i] * other.vals[i];
        }
        RowWise::<E> {
            vals: result,
            num_rows: self.num_rows,
        }
    }

    pub fn mul_inplace(&mut self, other: &RowWise<E>) {
        for i in 0..self.num_rows {
            self.vals[i] *= other.vals[i];
        }
    }

    pub fn max_inplace(&mut self, other: &RowWise<E>) {
        for i in 0..self.num_rows {
            self.vals[i] = max(self.vals[i], other.vals[i]);
        }
    }

    pub fn replace_at(&mut self, i: usize, new_val: E) {
        self.vals[i] = new_val;
    }

    pub fn cast_from<E2: Float>(row_wise: &RowWise<E>) -> RowWise<E2> {
        let num_rows = row_wise.num_rows;
        let mut vals = Array::<E2>::new(num_rows);

        for i in 0..num_rows {
            vals[i] = E2::cast_from(row_wise.vals[i]);
        }

        RowWise::<E2> { vals, num_rows }
    }
}

#[cube]
impl<E: Float> RowWise<E> {
    /// Per-row `e^(self - other)`.
    pub fn exp_diff(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut vals = Array::<E>::new(self.num_rows);

        for i in 0..self.num_rows {
            vals[i] = (self.vals[i] - other.vals[i]).exp();
        }

        RowWise::<E> {
            vals,
            num_rows: self.num_rows,
        }
    }

    /// `v -> 1/v` per row, with `v == 0` (fully-masked row) staying zero.
    pub fn recip_inplace(&mut self) {
        for i in 0..self.num_rows {
            let row_val = self.vals[i];

            let epsilon = E::new(FULLY_MASKED_ROW_THRESHOLD);
            let not_masked = E::cast_from(row_val >= epsilon);
            let safe_val = clamp_min(row_val, epsilon);
            let recip = safe_val.recip();
            self.vals[i] = not_masked * recip;
        }
    }
}
