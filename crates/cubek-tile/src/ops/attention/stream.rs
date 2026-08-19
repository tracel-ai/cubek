//! Register-resident attention for the decode shape: a few query rows, K/V
//! streamed straight through registers, no score tile, no shared memory, no
//! barriers until the ending.
//!
//! The fold is the online softmax: a per-row running `(max, sum)` state
//! ([`RowState`]) plus a value accumulator, updated one key position at a
//! time. Streaming means each K/V row is read from gmem once, folded in
//! immediately, and dropped. Plane-scoped by construction (a `plane_sum`
//! closes each dot): the cube's x dim must be exactly one plane; split teams
//! ride the y dim, each folding its own slice of the key positions.

use cubecl::prelude::*;

use crate::{instruction::plane, microkernel::horizontal, *};

/// One plane's share of the streamed fold.
///
/// Holds the plane's query rows and output accumulators in registers, at
/// lane-cyclic line ownership: lane `l` owns lines `l, l + lanes, ...` of
/// every row. The running [`RowState`] is replicated in every lane, since a
/// `plane_sum`ed score is plane-uniform.
#[derive(CubeType)]
pub struct StreamFold<EA: Float, N: Size> {
    q: Array<Vector<EA, N>>,
    acc: Array<Vector<EA, N>>,
    pub state: RowState<EA>,
    #[cube(comptime)]
    rows: usize,
    #[cube(comptime)]
    lines: usize,
    #[cube(comptime)]
    per_lane: usize,
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    width: usize,
}

#[cube]
impl<EA: Float, N: Size> StreamFold<EA, N> {
    /// Load the fold's operands off the final query tile: this lane's lines
    /// of every row, zeroed accumulators, the identity state.
    ///
    /// `q` is `{rows..., head_dim}`, group-major. `lanes` is the plane width
    /// (the cube's x dim). `N` must be bound to the operands' vector width
    /// (`let size!(N) = q.vector_size()`).
    pub fn new<EI: Numeric>(
        q: &Tile<EI>,
        #[comptime] lanes: usize,
        #[comptime] row_space: Space,
    ) -> StreamFold<EA, N> {
        let w = q.vector_size();
        let rank = comptime!(q.space.rank());
        let d = comptime!(q.space.extent_at(rank - 1));
        let rows = comptime!(row_space.tile_size());
        comptime!(assert!(
            q.space.tile_size() == rows * d && d.is_multiple_of(w),
            "StreamFold: q is {{rows..., head_dim}} with the line width dividing the head dim"
        ));
        let lines = comptime!(d / w);
        let per_lane = comptime!(lines.div_ceil(lanes));

        let qf = q.dense::<N>();
        let mut qa = Array::<Vector<EA, N>>::new(comptime!(rows * per_lane));
        let mut acc = Array::<Vector<EA, N>>::new(comptime!(rows * per_lane));
        let lane = UNIT_POS_X as usize;
        #[unroll]
        for g in 0..rows {
            #[unroll]
            for p in 0..per_lane {
                let li = lane + p * lanes;
                let mut qv = Vector::<EA, N>::cast_from(0u32);
                if li < lines {
                    qv = Vector::<EA, N>::cast_from(qf[g * lines + li]);
                }
                qa[g * per_lane + p] = qv;
                acc[g * per_lane + p] = Vector::<EA, N>::cast_from(0u32);
            }
        }

        StreamFold::<EA, N> {
            q: qa,
            acc,
            state: RowState::<EA>::new(row_space, 1usize),
            rows,
            lines,
            per_lane,
            lanes,
            width: comptime!(w),
        }
    }

    /// Fold one K/V block in: per key position, the lane-cooperative dot,
    /// one `plane_sum` per row, the state update, then the value row
    /// rescaled-and-accumulated. Positions at or past `cols_bound` (the
    /// ragged tail) are neither read nor folded, so the block need not
    /// divide anything.
    pub fn absorb<EI: Numeric>(
        &mut self,
        k: &Tile<EI>,
        v: &Tile<EI>,
        scale: EA,
        cols_bound: usize,
    ) {
        let wk = k.vector_size();
        let wv = v.vector_size();
        let rank = comptime!(k.space.rank());
        comptime!(assert!(
            wk == self.width && wv == self.width,
            "StreamFold: k and v share q's line width"
        ));
        comptime!(assert!(
            k.space.extent_at(rank - 1) == self.lines * self.width
                && v.space.extent_at(v.space.rank() - 1) == self.lines * self.width,
            "StreamFold: k and v trailing axes are the head dim"
        ));
        let cols = comptime!(k.space.extent_at(rank - 2));
        let rows = comptime!(self.rows);
        let lines = comptime!(self.lines);
        let per_lane = comptime!(self.per_lane);
        let lanes = comptime!(self.lanes);
        let w = comptime!(self.width);

        let kf = k.dense::<N>();
        let vf = v.dense::<N>();
        let lane = UNIT_POS_X as usize;
        let bound = min(cols_bound, cols);

        for s in 0..bound {
            let mut partial = Array::<EA>::new(rows);
            #[unroll]
            for g in 0..rows {
                partial[g] = EA::from_int(0);
            }
            #[unroll]
            for p in 0..per_lane {
                let li = lane + p * lanes;
                if li < lines {
                    let kv = Vector::<EA, N>::cast_from(kf[s * lines + li]);
                    #[unroll]
                    for g in 0..rows {
                        partial[g] += horizontal::vector::<EA, N>(
                            self.q[g * per_lane + p] * kv,
                            w,
                            LeafOp::Sum,
                        );
                    }
                }
            }

            let mut corrections = Array::<EA>::new(rows);
            let mut weights = Array::<EA>::new(rows);
            #[unroll]
            for g in 0..rows {
                let dot = plane::reduce::<EA>(partial[g], lanes, LeafOp::Sum);
                let score = dot * scale;
                let rescale = self.state.absorb(g, score);
                corrections[g] = rescale.correction;
                weights[g] = rescale.weight;
            }

            #[unroll]
            for p in 0..per_lane {
                let li = lane + p * lanes;
                if li < lines {
                    let vv = Vector::<EA, N>::cast_from(vf[s * lines + li]);
                    #[unroll]
                    for g in 0..rows {
                        self.acc[g * per_lane + p] = self.acc[g * per_lane + p]
                            * Vector::cast_from(corrections[g])
                            + vv * Vector::cast_from(weights[g]);
                    }
                }
            }
        }
    }

    /// The fused ending: every plane parks its state and accumulator lines
    /// in shared memory, one `sync_cube`, then plane 0 merges the states
    /// across splits and writes the normalized output, vectorized and cast
    /// to the output element. Fully-masked rows store exact zeros.
    pub fn store<EI: Numeric>(&self, out: &mut Tile<EI>, #[comptime] splits: usize) {
        let rows = comptime!(self.rows);
        let per_lane = comptime!(self.per_lane);
        let lanes = comptime!(self.lanes);
        let lines = comptime!(self.lines);
        let wo = out.vector_size();
        comptime!(assert!(
            wo == self.width,
            "StreamFold::store: the output shares the fold's line width"
        ));
        comptime!(assert!(
            out.space.tile_size() == rows * lines * self.width,
            "StreamFold::store: the output window is rows x head_dim"
        ));

        let mut partials =
            Shared::<[Vector<EA, N>]>::new_slice(comptime!(rows * splits * per_lane * lanes));
        let mut maxes = Shared::<[EA]>::new_slice(comptime!(rows * splits));
        let mut sums = Shared::<[EA]>::new_slice(comptime!(rows * splits));
        let lane = UNIT_POS_X as usize;
        let split = UNIT_POS_Y as usize;
        if lane == 0 {
            #[unroll]
            for g in 0..rows {
                maxes[g * splits + split] = self.state.m[g];
                sums[g * splits + split] = self.state.l[g];
            }
        }
        #[unroll]
        for g in 0..rows {
            #[unroll]
            for i in 0..per_lane {
                partials[((g * splits + split) * per_lane + i) * lanes + lane] =
                    self.acc[g * per_lane + i];
            }
        }
        sync_cube();

        // Plane 0 owns the merge and the store; the other planes are done.
        if split == 0 {
            let of = out.dense_mut::<N>();
            #[unroll]
            for g in 0..rows {
                let mut row_max = maxes[g * splits];
                for t in 1..splits {
                    row_max = max(row_max, maxes[g * splits + t]);
                }
                let mut weights = Array::<EA>::new(splits);
                let mut norm = EA::from_int(0);
                for t in 0..splits {
                    weights[t] = (maxes[g * splits + t] - row_max).exp();
                    norm += sums[g * splits + t] * weights[t];
                }
                let recip = masked_recip::<EA>(norm);
                #[unroll]
                for i in 0..per_lane {
                    let li = lane + i * lanes;
                    if li < lines {
                        let mut total = Vector::<EA, N>::cast_from(0u32);
                        for t in 0..splits {
                            total += partials[((g * splits + t) * per_lane + i) * lanes + lane]
                                * Vector::cast_from(weights[t]);
                        }
                        of[g * lines + li] = Vector::cast_from(total * Vector::cast_from(recip));
                    }
                }
            }
        }
    }

    /// The split ending: publish this plane's running state (lane 0 writes,
    /// the states are plane-uniform) and its accumulator lines into the
    /// team's windows of the split-wide buffers. The caller syncs, merges
    /// ([`Tile::merge_splits`]) and drains with the weights folded in.
    pub fn publish(&self, m_win: &mut Tile<EA>, l_win: &mut Tile<EA>, acc_win: &mut Tile<EA>) {
        let rows = comptime!(self.rows);
        m_win.store_rows(&self.state.m, rows);
        l_win.store_rows(&self.state.l, rows);

        let d = comptime!(self.lines * self.width);
        comptime!(assert!(
            acc_win.space.tile_size() == rows * d,
            "StreamFold::publish: the accumulator window is {{rows, head_dim}}"
        ));
        let per_lane = comptime!(self.per_lane);
        let lanes = comptime!(self.lanes);
        let lines = comptime!(self.lines);
        let w = comptime!(self.width);
        let wa = acc_win.vector_size();
        comptime!(assert!(
            wa == 1,
            "StreamFold::publish: a scalar accumulator window"
        ));
        let size!(W1) = wa;
        let mut af = acc_win.flat_mut::<W1>();
        let lane = UNIT_POS_X as usize;
        #[unroll]
        for g in 0..rows {
            #[unroll]
            for p in 0..per_lane {
                let li = lane + p * lanes;
                if li < lines {
                    #[unroll]
                    for j in 0..w {
                        af.write(
                            g * d + li * w + j,
                            Vector::cast_from(self.acc[g * per_lane + p].extract(j)),
                        );
                    }
                }
            }
        }
    }
}
