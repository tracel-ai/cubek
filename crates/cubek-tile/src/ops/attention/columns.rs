//! The attention verb's matmul leaves, at column ownership: each unit owns a
//! cyclic slice of the output's trailing axis, so a gmem operand streamed
//! along it is read exactly once across the owning team. Ownership is
//! team-local (`UNIT_POS_X` of `CUBE_DIM_X`): a split fold lays its teams on
//! the cube's y dim, each folding its own S range; an unsplit cube (y = 1)
//! spans every unit. Leaf-scoped like
//! [`softmax`](super::softmax): the verb owns the walk, the staging, and the
//! syncs; these run on already-lowered final tiles with no syncs of their own.
//!
//! Trailing-two-axes convention throughout (matmul's): leading degenerate axes
//! ride the flat index.

use cubecl::prelude::*;

use crate::*;

/// Comptime row chunk bounding the per-unit vector accumulators: `chunk × W`
/// registers live at once, whatever the score tile's height.
const ROW_CHUNK: usize = 4;

#[cube]
impl<EA: Float> Tile<EA> {
    /// `self[r, c] = dot(q[r, :], k[c, :])` — the score matmul, `self` a final
    /// rank-2 `{rows, cols}` memory tile. Each unit owns columns
    /// `UNIT_POS, UNIT_POS + CUBE_DIM, …` and streams whole `k` rows for
    /// them, so a gmem `k` is read once across the cube; `q` is read
    /// `cols`-fold and belongs in shared memory. Accumulates in `EA` lanes,
    /// horizontal-summed once per cell. `cols_bound` clips the ragged tail:
    /// columns at or past it are neither read from `k` (which may end before
    /// the block does) nor written — the softmax's mask probe overwrites
    /// every out-of-prefix cell regardless of its stale content.
    /// The caller syncs after.
    pub fn score_columns<EI: Numeric>(&mut self, q: &Tile<EI>, k: &Tile<EI>, cols_bound: usize) {
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

        let chunks = comptime!(rows.div_ceil(ROW_CHUNK));
        let workers = CUBE_DIM_X as usize;
        let bound = min(cols_bound, cols);
        let mut c = UNIT_POS_X as usize;
        while c < bound {
            #[unroll]
            for ch in 0..chunks {
                let base = comptime!(ch * ROW_CHUNK);
                let height = comptime!(ROW_CHUNK.min(rows - base));
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
                    let mut s = EA::from_int(0);
                    #[unroll]
                    for j in 0..wq {
                        s += acc[i].extract(j);
                    }
                    out.write((base + i) * cols + c, Vector::cast_from(s));
                }
            }
            c += workers;
        }
    }

    /// `self[r, v..] = self[r, v..] · factors[r] + Σ_{c < cols_bound}
    /// p[r, c] · val[c, v..]` — the value matmul with the online-softmax
    /// rescale fused in (each cell has one owner here, so the correction
    /// rides the same visit). `self` is a final rank-2 `{rows, val_dim}`
    /// accumulator; each unit owns cyclic *lines* of the value axis, so a
    /// gmem `val` is read once across the cube and adjacent units read
    /// adjacent lines (coalesced). `cols_bound` clips the block's ragged
    /// tail: stale cache beyond the attended prefix (possibly NaN) must not
    /// ride a zero probability. The caller syncs on both sides.
    pub fn mix_columns<EP: Numeric, EI: Numeric>(
        &mut self,
        p: &Tile<EP>,
        val: &Tile<EI>,
        factors: &Tile<EA>,
        cols_bound: usize,
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
        let chunks = comptime!(rows.div_ceil(ROW_CHUNK));
        let workers = CUBE_DIM_X as usize;
        let mut li = UNIT_POS_X as usize;
        while li < v_lines {
            #[unroll]
            for ch in 0..chunks {
                let base = comptime!(ch * ROW_CHUNK);
                let height = comptime!(ROW_CHUNK.min(rows - base));
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
