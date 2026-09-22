//! The online-softmax step at unit ownership. Each op runs over the rows a [`RowState`] owns
//! ([`owned_row`](RowState::owned_row)) with no syncs, a line (the tile's vector width of adjacent
//! columns) at a time, every loop over a comptime bound.

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
        state: &RowState<EA>,
    ) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            let r = state.owned_row(ri);
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

    /// Per owned row max into `acc`, seeded from the running max `state.m`.
    pub fn row_max(&self, acc: &mut Array<EA>, state: &RowState<EA>) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = state.m[ri];
            let r = state.owned_row(ri);
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
    pub fn exp_diff(&mut self, rowwise: &Array<EA>, state: &RowState<EA>) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            let r = state.owned_row(ri);
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
    pub fn row_sum(&self, acc: &mut Array<EA>, state: &RowState<EA>) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = EA::from_int(0);
            let r = state.owned_row(ri);
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
