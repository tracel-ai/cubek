//! Row-wise ops on a final tile: the legacy fragment row ops re-expressed on
//! tiles. Each op runs over the unit's owned rows (`rpu` contiguous rows per
//! unit, unit u starting at `u*rpu`) with no syncs. The backward's row ops
//! (prepass rowsum) join here.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// `self = self * scale`, masked entries driven to `min_value` (below the
    /// masked-logit threshold), per owned row.
    pub fn scale_and_mask(
        &mut self,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        #[comptime] rpu: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<W>();

        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                let q = probe.origin_q + r;
                for c in 0..cols {
                    let masked = probe.masked(q, probe.origin_s + c, mask);
                    let val = select(
                        masked,
                        EA::min_value(),
                        view.read(r * cols + c).extract(0) * scale,
                    );
                    view.write(r * cols + c, Vector::cast_from(val));
                }
            }
        }
    }

    /// Per owned row max into `acc`, seeded from `base`.
    pub fn row_max(&self, acc: &mut Array<EA>, base: &Array<EA>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let view = self.flat::<EA, W>();

        for ri in 0..rpu {
            acc[ri] = base[ri];
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                for c in 0..cols {
                    acc[ri] = max(acc[ri], view.read(r * cols + c).extract(0));
                }
            }
        }
    }

    /// `self = exp(self - rowwise)` per owned row, with the fully-masked
    /// guard: a row whose max is below the threshold goes entirely to zero.
    pub fn exp_diff(&mut self, rowwise: &Array<EA>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<W>();

        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                for c in 0..cols {
                    let e = live * (view.read(r * cols + c).extract(0) - safe_m).exp();
                    view.write(r * cols + c, Vector::cast_from(e));
                }
            }
        }
    }

    /// Per owned row sum into `acc`.
    pub fn row_sum(&self, acc: &mut Array<EA>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let view = self.flat::<EA, W>();

        for ri in 0..rpu {
            acc[ri] = EA::from_int(0);
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                for c in 0..cols {
                    acc[ri] += view.read(r * cols + c).extract(0);
                }
            }
        }
    }

    /// Cast-copy the owned rows into `dest`.
    pub fn write_rows_to<EP: Numeric>(&self, dest: &mut Tile<EP>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let size!(WP) = dest.vector_size();
        let src = self.flat::<EA, W>();
        let mut dst = dest.flat_mut::<WP>();

        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                for c in 0..cols {
                    dst.write(
                        r * cols + c,
                        Vector::cast_from(src.read(r * cols + c).extract(0)),
                    );
                }
            }
        }
    }
}
