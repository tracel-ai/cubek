//! The online-softmax step at unit ownership: the legacy fragment row ops
//! re-expressed on tiles. Each op runs over the unit's owned rows (`rpu`
//! contiguous rows per unit, unit u starting at `u*rpu`) with no syncs, a line
//! — the tile's vector width of adjacent columns — at a time, every loop over a
//! comptime bound. The backward's row ops (prepass rowsum) join here.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// `self = self * scale`, masked entries driven to `min_value` (below the
    /// masked-logit threshold), per owned row. A row is read and written a
    /// line at a time, the tile's vector width of adjacent columns.
    pub fn scale_and_mask(
        &mut self,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        #[comptime] rpu: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                let q = probe.row_q(r);
                #[unroll]
                for line in 0..lines {
                    let i = r * lines + line;
                    let mut v = view.read(i) * Vector::<EA, W>::cast_from(scale);
                    #[unroll]
                    for j in 0..w {
                        let masked = probe.masked(q, probe.origin_s + line * w + j, mask);
                        v.insert(j, select(masked, EA::min_value(), v.extract(j)));
                    }
                    view.write(i, v);
                }
            }
        }
    }

    /// [`scale_and_mask`](Tile::scale_and_mask) and
    /// [`row_max`](Tile::row_max) in one sweep: the max is taken over the value
    /// just scaled and masked, before it is written back, so the tile is read
    /// once instead of twice.
    pub fn scale_mask_max(
        &mut self,
        acc: &mut Array<EA>,
        base: &Array<EA>,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        #[comptime] rpu: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = base[ri];
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                let q = probe.row_q(r);
                #[unroll]
                for line in 0..lines {
                    let i = r * lines + line;
                    let mut v = view.read(i) * Vector::<EA, W>::cast_from(scale);
                    #[unroll]
                    for j in 0..w {
                        let masked = probe.masked(q, probe.origin_s + line * w + j, mask);
                        let cell = select(masked, EA::min_value(), v.extract(j));
                        v.insert(j, cell);
                        acc[ri] = max(acc[ri], cell);
                    }
                    view.write(i, v);
                }
            }
        }
    }

    /// [`exp_diff`](Tile::exp_diff) and [`row_sum`](Tile::row_sum) in one
    /// sweep: the sum accumulates the probabilities just written, so the tile
    /// is read once instead of twice.
    pub fn exp_diff_sum(
        &mut self,
        acc: &mut Array<EA>,
        rowwise: &Array<EA>,
        #[comptime] rpu: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = EA::from_int(0);
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                #[unroll]
                for line in 0..lines {
                    let i = r * lines + line;
                    let mut v = view.read(i);
                    #[unroll]
                    for j in 0..w {
                        let cell = live * (v.extract(j) - safe_m).exp();
                        v.insert(j, cell);
                        acc[ri] += cell;
                    }
                    view.write(i, v);
                }
            }
        }
    }

    /// Per owned row max into `acc`, seeded from `base`.
    pub fn row_max(&self, acc: &mut Array<EA>, base: &Array<EA>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = base[ri];
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                #[unroll]
                for line in 0..lines {
                    let v = view.read(r * lines + line);
                    #[unroll]
                    for j in 0..w {
                        acc[ri] = max(acc[ri], v.extract(j));
                    }
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
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                #[unroll]
                for line in 0..lines {
                    let i = r * lines + line;
                    let mut v = view.read(i);
                    #[unroll]
                    for j in 0..w {
                        v.insert(j, live * (v.extract(j) - safe_m).exp());
                    }
                    view.write(i, v);
                }
            }
        }
    }

    /// Per owned row sum into `acc`.
    pub fn row_sum(&self, acc: &mut Array<EA>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = EA::from_int(0);
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                #[unroll]
                for line in 0..lines {
                    let v = view.read(r * lines + line);
                    #[unroll]
                    for j in 0..w {
                        acc[ri] += v.extract(j);
                    }
                }
            }
        }
    }
}
