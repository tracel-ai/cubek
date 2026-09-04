//! Row-wise ops on a final tile: the legacy fragment row ops re-expressed on
//! tiles. Each op runs over the unit's owned rows (`rpu` contiguous rows per
//! unit, unit u starting at `u*rpu`) with no syncs, a line — the tile's vector
//! width of adjacent columns — at a time, every loop over a comptime bound.
//! The backward's row ops (prepass rowsum) join here.

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

    /// Merge a split fold's per-team running states into cross-split weights:
    /// per row, `self[r, t] = exp(m[r, t] - m*) / Σ_t l[r, t] · exp(m[r, t] - m*)`
    /// where `m* = max_t m[r, t]`.
    ///
    /// The normalizer is folded into the weights, so `Σ_t self[r, t] · acc[r, t, ·]`
    /// is the whole merge: a drain that contracts the weights as an operand has
    /// no value in hand left to scale.
    ///
    /// `self`, `m` and `l` span the score rows and `split`, laid out alike. The
    /// space says where `split` sits: outermost gives a team one contiguous run
    /// of rows, innermost lets the drain contract it as a matmul's `k`.
    ///
    /// A fully-masked row gets weights of exactly zero, and a split that folded
    /// nothing published `(min, 0)` so it weighs zero on its own. One unit per
    /// row, cyclic over the cube; the caller syncs on both sides. A single split
    /// degenerates to the plain epilogue.
    pub fn merge_splits(&mut self, m: &Tile<EA>, l: &Tile<EA>, #[comptime] split: Axis) {
        let space = comptime!(self.space.clone());
        comptime!(assert!(
            space.contains(split),
            "merge_splits: {split:?} is not an axis of the weights"
        ));
        let splits = comptime!(space.extent(split));
        let rows = comptime!(space.tile_size() / splits);
        // The states can arrive partitioned differently from the weights (a
        // global buffer beside a shared-memory tile); only the layout has to agree.
        let m_space = comptime!(m.space.clone());
        let l_space = comptime!(l.space.clone());
        comptime!(assert!(
            m_space.laid_out_like(&space) && l_space.laid_out_like(&space),
            "merge_splits: the states must be laid out like the weights they merge into"
        ));

        // A step of `t` crosses whatever sits inside the split axis; a step of
        // the rows outside it crosses a whole slice of splits.
        let stride = comptime!(
            ((space.position(split) + 1)..space.rank())
                .map(|p| space.extent_at(p))
                .product::<usize>()
        );
        let slice = comptime!(splits * stride);

        let size!(W) = self.vector_size();
        let size!(WM) = m.vector_size();
        let size!(WL) = l.vector_size();
        let mf = m.flat::<WM>();
        let lf = l.flat::<WL>();
        let mut wf = self.flat_mut::<W>();

        let workers = CUBE_DIM as usize;
        let mut r = UNIT_POS as usize;
        while r < rows {
            // Row `r`'s first cell; its splits step by `stride` from there.
            let base = (r / stride) * slice + r % stride;
            let mut mstar = EA::min_value();
            for t in 0..splits {
                mstar = max(mstar, mf.read(base + t * stride).extract(0usize));
            }
            // The normalizer needs every split, so park the unnormalized
            // weights and scale them where they sit.
            let mut lstar = EA::from_int(0);
            for t in 0..splits {
                let w = (mf.read(base + t * stride).extract(0usize) - mstar).exp();
                lstar += lf.read(base + t * stride).extract(0usize) * w;
                wf.write(base + t * stride, Vector::cast_from(w));
            }
            let recip = masked_recip::<EA>(lstar);
            for t in 0..splits {
                let w = wf.read(base + t * stride).extract(0usize) * recip;
                wf.write(base + t * stride, Vector::cast_from(w));
            }
            r += workers;
        }
    }

    /// Publish per-owned-row `values` into this factors tile, one cell per
    /// score row. The caller syncs before any cross-worker read.
    ///
    /// Takes the [`RowShare`] rather than a count, because it has to agree with
    /// the leaf about who owns row `r` — and under
    /// [`Plane`](RowShare::Plane) it also has to write each row once, where
    /// every lane of the plane holds the value.
    ///
    /// A row lane at any rank: a fold's window on a split-wide tile is
    /// `{1, rows}`, the same cells as a plain `{rows}`.
    pub fn store_rows(&mut self, values: &Array<EA>, #[comptime] share: RowShare) {
        let rpu = comptime!(share.rows());
        let lanes = comptime!(share.lanes());
        let rows = comptime!(self.space.tile_size());
        comptime!(assert!(
            (0..self.space.rank())
                .filter(|&p| self.space.extent_at(p) > 1)
                .count()
                <= 1,
            "store_rows: a row lane, one cell per score row; this tile spans {:?}",
            self.space
        ));
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<W>();

        // One writer per row: the owning unit, or the owning plane's first
        // lane — every other lane holds the same value and writing it again
        // would be a race for nothing.
        let writer = (UNIT_POS_X as usize).is_multiple_of(lanes);
        #[unroll]
        for ri in 0..rpu {
            let r = (UNIT_POS_X as usize / lanes) * rpu + ri;
            if r < rows && writer {
                view.write(r, Vector::cast_from(values[ri]));
            }
        }
    }

    /// The online-softmax rescale, by the row's owner: `self[r, :] *= corr[ri]` for the rows
    /// `share` gives this worker, straight out of [`softmax`](Tile::softmax) and before the sync
    /// that hands the accumulator to the value matmul. The correction is the worker's own
    /// register — no factors tile, no cube-wide sweep, no barrier of its own. Under
    /// [`Plane`](RowShare::Plane) the lanes split the row's lines.
    pub fn rescale_rows(&mut self, corr: &Array<EA>, #[comptime] share: RowShare) {
        let rank = comptime!(self.space.rank());
        let rows = comptime!(self.space.extent_at(rank - 2));
        let cols = comptime!(self.space.extent_at(rank - 1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let rpw = comptime!(share.rows());
        let lanes = comptime!(share.lanes());
        let lane = UNIT_POS_X as usize % lanes;
        let worker = UNIT_POS_X as usize / lanes;
        let mut view = self.flat_mut::<W>();
        #[unroll]
        for ri in 0..rpw {
            let r = worker * rpw + ri;
            if r < rows {
                let factor = Vector::<EA, W>::cast_from(corr[ri]);
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(lanes)) {
                    let line = lane + li * lanes;
                    if comptime!(lines % lanes == 0) || line < lines {
                        let i = r * lines + line;
                        view.write(i, view.read(i) * factor);
                    }
                }
            }
        }
    }

    /// Multiply each row by its factor: `self[r, c] *= factors[r]`.
    ///
    /// The accumulator rescale between fold steps, and the epilogue
    /// normalize when the factors are `recip_l`. Cyclic over the whole cube
    /// so each cell is touched exactly once, whatever ownership the
    /// interleaved matmuls use; the caller syncs on both sides.
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
            let v = view.read(i).extract(0usize) * f.read(i / cols).extract(0usize);
            view.write(i, Vector::cast_from(v));
            i += workers;
        }
    }

    /// Cast-copy the owned rows into `dest`, which is laid out in the same
    /// lines.
    pub(crate) fn write_rows_to<EP: Numeric>(&self, dest: &mut Tile<EP>, #[comptime] rpu: usize) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let w = self.vector_size();
        let wp = dest.vector_size();
        comptime!(assert!(
            w == wp,
            "write_rows_to: the probabilities are laid out in the score's lines"
        ));
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let src = self.flat::<W>();
        let mut dst = dest.flat_mut::<W>();

        #[unroll]
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                #[unroll]
                for line in 0..lines {
                    let i = r * lines + line;
                    dst.write(i, Vector::<EP, W>::cast_from(src.read(i)));
                }
            }
        }
    }
}
