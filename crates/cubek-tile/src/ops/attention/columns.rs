//! Attention's matmul leaves at column ownership, the arm the software instruction runs: each
//! unit owns every `CUBE_DIM_X`-th column of the output, so a K or V block streamed along that
//! axis is read from gmem once per team. Split teams sit on the cube's y dim; a cube with y = 1
//! is one team spanning every unit. The hardware twin, where the worker is a plane and the visit
//! a fragment, is [`fragments`](super::fragments).
//!
//! Reached through [`score`](crate::Tile::score) and [`mix`](crate::Tile::mix), which read the
//! instruction off the accumulator's space. Trailing-two-axes convention
//! (matmul's): leading degenerate axes ride the flat index.

use cubecl::prelude::*;

use crate::{instruction::registers::horizontal, *};

#[cube]
impl<EA: Float> Tile<EA> {
    /// [`score`](Tile::score) under the software instruction. Each unit streams whole `k` rows
    /// for its owned columns, so a gmem `k` is read once per team; `q` is read `cols` times over
    /// and belongs in shared memory.
    pub(crate) fn score_columns<EI: Numeric>(
        &mut self,
        q: &Tile<EI>,
        k: &Tile<EI>,
        cols_bound: usize,
        #[comptime] config: RegisterBlock,
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
        let row_chunk = comptime!(rows_per_visit(config, wq, rows));
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
                    let s = horizontal::vector::<EA, WI>(acc[i], wq, Monoid::Sum);
                    out.write((base + i) * cols + c, Vector::cast_from(s));
                }
            }
            c += workers;
        }
    }

    /// [`mix`](Tile::mix) under the software instruction. A unit owns one `(row chunk, value
    /// line)` pair at a time, cyclically: the line is the inner digit, so adjacent units read
    /// adjacent lines of `val` and a gmem `val` is read once per team, coalesced. The rows are the
    /// other digit because a leaf spread over the value lines alone would sit entirely on the axis
    /// vectorization divides, where widening starves the grid and narrowing starves the bus.
    ///
    /// The rescale rides the same visit because each cell has exactly one owner here.
    pub(crate) fn mix_columns<EP: Numeric, EI: Numeric>(
        &mut self,
        p: &Tile<EP>,
        val: &Tile<EI>,
        factors: &Tile<EA>,
        cols_bound: usize,
        #[comptime] config: RegisterBlock,
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
        let row_chunk = comptime!(rows_per_visit(config, wv, rows));
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
        let height = comptime!(row_chunk);
        // One visit is a `(row chunk, value line)` pair; the line is the inner digit, so the
        // units sharing a chunk read consecutive lines.
        let visits = comptime!((rows / row_chunk) * v_lines);
        let workers = CUBE_DIM_X as usize;
        let mut visit = UNIT_POS_X as usize;
        while visit < visits {
            let base = (visit / v_lines) * row_chunk;
            let li = visit % v_lines;
            let mut acc = Array::<Vector<EA, WV>>::new(height);
            #[unroll]
            for i in 0..height {
                acc[i] = Vector::<EA, WV>::cast_from(0u32);
            }
            for c in 0..bound {
                let vv = Vector::<EA, WV>::cast_from(vf.read(c * v_lines + li));
                #[unroll]
                for i in 0..height {
                    let prob = EA::cast_from(pf.read((base + i) * cols + c).extract(0usize));
                    acc[i] += Vector::<EA, WV>::cast_from(prob) * vv;
                }
            }
            #[unroll]
            for i in 0..height {
                let f = ff.read(base + i).extract(0usize);
                #[unroll]
                for j in 0..wv {
                    let idx = (base + i) * val_dim + li * wv + j;
                    let cur = out.read(idx).extract(0usize);
                    out.write(idx, Vector::cast_from(cur * f + acc[i].extract(j)));
                }
            }
            visit += workers;
        }
    }
}

/// How many rows one visit keeps live, out of the instruction's [`budget`](RegisterBlock::budget)
/// of accumulator registers: the visit holds one `width`-wide line per row, so a wider line buys
/// fewer rows, not more registers.
///
/// The largest such count that also *divides* `rows`, so every visit is the same shape and none
/// straddles the last row: a visit is what a worker picks up, and a ragged one has no comptime
/// height to unroll over. Never zero, since a visit holding no row contracts nothing.
fn rows_per_visit(config: RegisterBlock, width: usize, rows: usize) -> usize {
    let cap = (config.budget / width).max(1).min(rows);
    (1..=cap)
        .rev()
        .find(|c| rows.is_multiple_of(*c))
        .unwrap_or(1)
}
