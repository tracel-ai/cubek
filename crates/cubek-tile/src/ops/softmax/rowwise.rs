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
                let q = probe.row_q(r);
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
        let view = self.flat::<W>();

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
        let view = self.flat::<W>();

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

    /// Merge a split fold's per-team running states into cross-split weights:
    /// `self` is the `{splits · rows}` weight tile, `m`/`l` the published
    /// per-split running max/sum (same shape), `recip` the `{rows}`
    /// inverse-sum tile. Per row: `m* = max_t m_t`, each split's weight is
    /// `exp(m_t − m*)`, `l* = Σ_t l_t · exp(m_t − m*)`, and `recip = 1/l*`
    /// (exactly zero for a fully-masked row, so its output stays zero). A
    /// split that folded nothing publishes `(min, 0)` and gets weight zero on
    /// its own. One unit per row, cyclic across the whole cube; the caller
    /// syncs on both sides. `splits == 1` degenerates to the plain epilogue.
    pub fn merge_splits(
        &mut self,
        recip: &mut Tile<EA>,
        m: &Tile<EA>,
        l: &Tile<EA>,
        #[comptime] splits: usize,
    ) {
        let total = comptime!(self.space.extent_at(0));
        let rows = comptime!(total / splits);
        comptime!(assert!(
            self.space.rank() == 1 && total.is_multiple_of(splits),
            "merge_splits: a rank-1 {{splits · rows}} weight tile"
        ));
        let size!(W) = self.vector_size();
        let size!(WR) = recip.vector_size();
        let size!(WM) = m.vector_size();
        let size!(WL) = l.vector_size();
        let mf = m.flat::<WM>();
        let lf = l.flat::<WL>();
        let mut rf = recip.flat_mut::<WR>();
        let mut wf = self.flat_mut::<W>();

        let workers = CUBE_DIM as usize;
        let mut r = UNIT_POS as usize;
        while r < rows {
            let mut mstar = EA::min_value();
            for t in 0..splits {
                mstar = max(mstar, mf.read(t * rows + r).extract(0));
            }
            let mut lstar = EA::from_int(0);
            for t in 0..splits {
                let w = (mf.read(t * rows + r).extract(0) - mstar).exp();
                lstar += lf.read(t * rows + r).extract(0) * w;
                wf.write(t * rows + r, Vector::cast_from(w));
            }
            let eps = EA::new(FULLY_MASKED_ROW_THRESHOLD);
            let rv = EA::cast_from(lstar >= eps) * clamp_min(lstar, eps).recip();
            rf.write(r, Vector::cast_from(rv));
            r += workers;
        }
    }

    /// Publish per-owned-row `values` into this rank-1 factors tile, one cell
    /// per row of the score space. Owners write their own rows only; the caller
    /// syncs before any cross-unit read.
    pub fn store_rows(&mut self, values: &Array<EA>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        comptime!(assert!(
            self.space.rank() == 1,
            "store_rows: a rank-1 factors tile, one cell per score row"
        ));
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<W>();

        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                view.write(r, Vector::cast_from(values[ri]));
            }
        }
    }

    /// `self[r, c] *= factors[r]` over the whole rank-2 tile: the accumulator
    /// rescale between fold steps (and the epilogue normalize, with `recip_l`
    /// factors). Cyclic across the cube so each cell is touched exactly once,
    /// whatever ownership the interleaved matmuls use; the caller syncs on both
    /// sides.
    pub fn scale_rows(&mut self, factors: &Tile<EA>) {
        let cols = comptime!(self.space.extent_at(1));
        comptime!(assert!(
            self.space.rank() == 2,
            "scale_rows: a rank-2 accumulator tile"
        ));
        let w = self.vector_size();
        let wf = factors.vector_size();
        comptime!(assert!(
            w == 1 && wf == 1,
            "scale_rows: vectorized tiles not supported yet"
        ));
        let total = comptime!(self.space.tile_size());
        let size!(W) = w;
        let size!(WF) = wf;
        let f = factors.flat::<WF>();
        let mut view = self.flat_mut::<W>();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let v = view.read(i).extract(0) * f.read(i / cols).extract(0);
            view.write(i, Vector::cast_from(v));
            i += workers;
        }
    }

    /// Cast-copy the owned rows into `dest`.
    pub fn write_rows_to<EP: Numeric>(&self, dest: &mut Tile<EP>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let size!(W) = self.vector_size();
        let size!(WP) = dest.vector_size();
        let src = self.flat::<W>();
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
