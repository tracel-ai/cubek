//! Attention's matmul leaves at column ownership: each unit owns every
//! `CUBE_DIM_X`-th column of the output, so a K or V block streamed along
//! that axis is read from gmem once per team. Split teams sit on the cube's
//! y dim; a cube with y = 1 is one team spanning every unit. Like
//! [`softmax`](crate::Tile::softmax) these are leaf ops on final tiles: the
//! caller owns the walk and the syncs. Trailing-two-axes convention
//! (matmul's): leading degenerate axes ride the flat index.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// The score matmul: `self[r, c] = dot(q[r, :], k[c, :])`.
    ///
    /// `self` is a final rank-2 `{rows, cols}` scalar tile. Each unit streams
    /// whole `k` rows for its owned columns, so a gmem `k` is read once per
    /// team; `q` is read `cols` times over and belongs in shared memory.
    /// Columns at or past `cols_bound` are neither read nor written: `k` may
    /// end before the block does, and the softmax's mask probe overwrites
    /// those cells anyway. `row_chunk` caps the live accumulators (that many
    /// vectors at once), a register-budget decision the caller makes.
    /// The caller syncs after.
    pub fn score_columns<EI: Numeric>(
        &mut self,
        q: &Tile<EI>,
        k: &Tile<EI>,
        cols_bound: usize,
        #[comptime] row_chunk: usize,
    ) {
        let rank = comptime!(self.space.rank());
        let rows = comptime!(self.space.extent_at(rank - 2));
        let cols = comptime!(self.space.extent_at(rank - 1));
        let d = comptime!(q.space.extent_at(q.space.rank() - 1));
        let w = self.vector_size();
        let wq = q.vector_size();
        let wk = k.vector_size();
        comptime!(assert!(
            w == 1,
            "score_columns: a scalar score tile (softmax's contract)"
        ));
        comptime!(assert!(
            wq == wk && d.is_multiple_of(wq),
            "score_columns: q and k share one line width dividing the head dim"
        ));
        comptime!(assert!(
            k.space.extent_at(k.space.rank() - 1) == d,
            "score_columns: k's trailing axis is the contracted head dim"
        ));
        let lines = comptime!(d / wq);
        let size!(W) = w;
        let size!(WI) = wq;

        let qf = q.flat::<WI>();
        let kf = k.flat::<WI>();
        let mut out = self.flat_mut::<W>();

        let chunks = comptime!(rows.div_ceil(row_chunk));
        let workers = CUBE_DIM_X as usize;
        let bound = min(cols_bound, cols);
        let mut c = UNIT_POS_X as usize;
        while c < bound {
            #[unroll]
            for ch in 0..chunks {
                let base = comptime!(ch * row_chunk);
                let height = comptime!(row_chunk.min(rows - base));
                let mut acc = Array::<Vector<EA, WI>>::new(height);
                #[unroll]
                for i in 0..height {
                    acc[i] = Vector::<EA, WI>::cast_from(0u32);
                }
                for l in 0..lines {
                    let kv = Vector::<EA, WI>::cast_from(kf.read(c * lines + l));
                    #[unroll]
                    for i in 0..height {
                        let qv = Vector::<EA, WI>::cast_from(qf.read((base + i) * lines + l));
                        acc[i] += qv * kv;
                    }
                }
                #[unroll]
                for i in 0..height {
                    let s = hsum(acc[i], wq);
                    out.write((base + i) * cols + c, Vector::cast_from(s));
                }
            }
            c += workers;
        }
    }

    /// The value matmul with the online-softmax rescale fused in:
    /// `self[r, :] = self[r, :] · factors[r] + Σ_{c < cols_bound} p[r, c] · val[c, :]`.
    ///
    /// `self` is a final rank-2 `{rows, val_dim}` accumulator. Each unit owns
    /// every `CUBE_DIM_X`-th line of the value axis, so a gmem `val` is read
    /// once per team and adjacent units read adjacent lines. The rescale
    /// rides the same visit because each cell has exactly one owner here.
    /// Columns at or past `cols_bound` are skipped: stale cache beyond the
    /// attended prefix (possibly NaN) must not ride a zero probability.
    /// `row_chunk` as in [`score_columns`](Tile::score_columns). The caller
    /// syncs on both sides.
    pub fn mix_columns<EP: Numeric, EI: Numeric>(
        &mut self,
        p: &Tile<EP>,
        val: &Tile<EI>,
        factors: &Tile<EA>,
        cols_bound: usize,
        #[comptime] row_chunk: usize,
    ) {
        let rank = comptime!(self.space.rank());
        let rows = comptime!(self.space.extent_at(rank - 2));
        let val_dim = comptime!(self.space.extent_at(rank - 1));
        let cols = comptime!(p.space.extent_at(p.space.rank() - 1));
        let w = self.vector_size();
        let wp = p.vector_size();
        let wv = val.vector_size();
        comptime!(assert!(
            w == 1 && wp == 1,
            "mix_columns: scalar accumulator and probability tiles"
        ));
        comptime!(assert!(
            val_dim.is_multiple_of(wv),
            "mix_columns: val's line width divides the value dim"
        ));
        let v_lines = comptime!(val_dim / wv);
        let size!(W) = w;
        let size!(WP) = wp;
        let size!(WV) = wv;

        let wf = factors.vector_size();
        comptime!(assert!(wf == 1, "mix_columns: a scalar factors tile"));
        let size!(WF) = wf;
        let pf = p.flat::<WP>();
        let vf = val.flat::<WV>();
        let ff = factors.flat::<WF>();
        let mut out = self.flat_mut::<W>();

        let bound = min(cols_bound, cols);
        let chunks = comptime!(rows.div_ceil(row_chunk));
        let workers = CUBE_DIM_X as usize;
        let mut li = UNIT_POS_X as usize;
        while li < v_lines {
            #[unroll]
            for ch in 0..chunks {
                let base = comptime!(ch * row_chunk);
                let height = comptime!(row_chunk.min(rows - base));
                let mut acc = Array::<Vector<EA, WV>>::new(height);
                #[unroll]
                for i in 0..height {
                    acc[i] = Vector::<EA, WV>::cast_from(0u32);
                }
                for c in 0..bound {
                    let vv = Vector::<EA, WV>::cast_from(vf.read(c * v_lines + li));
                    #[unroll]
                    for i in 0..height {
                        let prob = EA::cast_from(pf.read((base + i) * cols + c).extract(0));
                        acc[i] += Vector::<EA, WV>::cast_from(prob) * vv;
                    }
                }
                #[unroll]
                for i in 0..height {
                    let f = ff.read(base + i).extract(0);
                    #[unroll]
                    for j in 0..wv {
                        let idx = (base + i) * val_dim + li * wv + j;
                        let cur = out.read(idx).extract(0);
                        out.write(idx, Vector::cast_from(cur * f + acc[i].extract(j)));
                    }
                }
            }
            li += workers;
        }
    }
}

/// Horizontal sum of a vector's `width` lanes.
#[cube]
pub(super) fn hsum<E: Float, N: Size>(v: Vector<E, N>, #[comptime] width: usize) -> E {
    let mut s = E::from_int(0);
    #[unroll]
    for j in 0..width {
        s += v.extract(j);
    }
    s
}
