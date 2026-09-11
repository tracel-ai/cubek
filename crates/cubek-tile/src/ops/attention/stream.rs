//! Register-resident attention for the decode shape: a few query rows, K/V
//! streamed straight through registers, no score tile, no shared memory, no
//! barriers until the ending.
//!
//! The fold is the online softmax: a per-row running `(max, sum)` state
//! ([`RowState`]) plus a value accumulator. Streaming means each K/V row is
//! read from gmem once, folded in immediately, and dropped. Plane-scoped by
//! construction (a lane butterfly closes each dot): the cube's x dim must be
//! exactly one plane; split teams ride the y dim, each folding its own slice of
//! the key positions.
//!
//! # Teams within the plane
//!
//! A plane does not spend all its lanes on one key position. It splits into
//! `lanes / span` **teams** of `span` lanes, each folding every
//! `teams`-th position of the block with a state of its own, and the teams
//! meet once, at the ending, the way split planes do. The span is sized so a
//! lane holds a few lines of the head rather than one, which is what the walk
//! is bound by on a GPU: instructions per position, not bytes.
//!
//! A plane on one position runs one butterfly per score — five shuffles on a
//! 32-wide plane, one position's worth of work each — and at a 128-wide f16
//! head leaves half its lanes idle (16 lines of 8). On GP100 that walked a
//! 2245-position head at 20 GB/s. A team of 8 lanes closes its score in three
//! shuffles that four teams issue together, and every lane is loaded.

use cubecl::prelude::*;

use crate::{instruction::plane, instruction::registers::horizontal, *};

/// Key positions one team folds per step of [`StreamFold::absorb`] — the loads
/// it puts in flight before the first is used, and the independent reductions
/// it hands the scheduler.
const CHUNK: usize = 4;

/// Floats of query and accumulator a lane is sized to hold: what the span is
/// chosen against, so the lines a lane owns grow as the rows shrink. Past it
/// the fold would spill before it gained anything.
const LANE_FLOATS: usize = 64;

/// Lanes on one key position: enough to hold a head's lines at
/// [`LANE_FLOATS`] per lane, a power of two so a team is an aligned run of
/// lanes a butterfly closes without leaving it, and never wider than the plane.
fn team_span(lines: usize, lanes: usize, rows: usize, width: usize) -> usize {
    if lanes <= 1 || !lanes.is_power_of_two() {
        return lanes.max(1);
    }
    let per_lane = (LANE_FLOATS / (2 * rows * width)).max(1);
    lines.div_ceil(per_lane).next_power_of_two().min(lanes)
}

/// One plane's share of the streamed fold.
///
/// Holds the plane's query rows and output accumulators in registers. The
/// plane splits into teams of `span` lanes (see the module doc): lane `l` is
/// team `l / span` and owns lines `l % span, l % span + span, ...` of every
/// row, and each team carries a running [`RowState`] of its own, replicated
/// across its lanes, until [`store`](Self::store) or
/// [`publish`](Self::publish) closes the teams into one.
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
    span: usize,
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
        let span = comptime!(team_span(lines, lanes, rows, w));
        let per_lane = comptime!(lines.div_ceil(span));

        let qf = q.dense::<N>();
        let mut qa = Array::<Vector<EA, N>>::new(comptime!(rows * per_lane));
        let mut acc = Array::<Vector<EA, N>>::new(comptime!(rows * per_lane));
        let first = UNIT_POS_X as usize % span;
        #[unroll]
        for g in 0..rows {
            #[unroll]
            for p in 0..per_lane {
                let li = first + p * span;
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
            span,
            width: comptime!(w),
        }
    }

    /// Fold one K/V block in. Each team takes every `teams`-th position,
    /// [`CHUNK`] of them per step: their K and V lines loaded first, then the
    /// dots, each closed across the team's lanes, one state update for the
    /// chunk, and the value rows rescaled-and-accumulated. Positions at or
    /// past `cols_bound` (the ragged tail) are neither read nor folded, so the
    /// block need not divide anything.
    ///
    /// **The chunk is what keeps the walk from being latency-bound.** One
    /// position at a time, each step is a dependent chain — load K, reduce,
    /// update the state, load V — with one position's lines in flight. A chunk
    /// puts every load of the step in flight before the first is used, and its
    /// reductions and exponentials are independent chains the scheduler
    /// interleaves.
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
        let span = comptime!(self.span);
        let teams = comptime!(self.lanes / self.span);
        let w = comptime!(self.width);
        let chunk = comptime!(CHUNK.min(cols.div_ceil(teams)).max(1));

        let kf = k.dense::<N>();
        let vf = v.dense::<N>();
        let lane = UNIT_POS_X as usize;
        let team = lane / span;
        let first = lane % span;
        let bound = min(cols_bound, cols);

        let mut s0 = 0usize;
        while s0 < bound {
            // Every load of the step issued before any is used. A position
            // past the bound reads nothing and holds zeros, which its zero
            // weight below keeps out of the mix.
            let mut ks = Array::<Vector<EI, N>>::new(comptime!(chunk * per_lane));
            let mut vs = Array::<Vector<EI, N>>::new(comptime!(chunk * per_lane));
            #[unroll]
            for c in 0..chunk {
                let s = s0 + c * teams + team;
                #[unroll]
                for p in 0..per_lane {
                    let li = first + p * span;
                    let mut kv = Vector::<EI, N>::cast_from(0u32);
                    let mut vv = Vector::<EI, N>::cast_from(0u32);
                    if li < lines && s < bound {
                        kv = kf[s * lines + li];
                        vv = vf[s * lines + li];
                    }
                    ks[c * per_lane + p] = kv;
                    vs[c * per_lane + p] = vv;
                }
            }

            let mut scores = Array::<EA>::new(comptime!(rows * chunk));
            #[unroll]
            for g in 0..rows {
                #[unroll]
                for c in 0..chunk {
                    let mut partial = EA::from_int(0);
                    #[unroll]
                    for p in 0..per_lane {
                        let li = first + p * span;
                        if li < lines {
                            partial += horizontal::vector::<EA, N>(
                                self.q[g * per_lane + p]
                                    * Vector::<EA, N>::cast_from(ks[c * per_lane + p]),
                                w,
                                Monoid::Sum,
                            );
                        }
                    }
                    scores[g * chunk + c] = team_fold::<EA>(partial, span, Monoid::Sum) * scale;
                }
            }

            // One state update per chunk: its maximum folds in once, so the
            // accumulator is rescaled once rather than per position.
            let mut corrections = Array::<EA>::new(rows);
            let mut weights = Array::<EA>::new(comptime!(rows * chunk));
            #[unroll]
            for g in 0..rows {
                let m_old = self.state.m[g];
                let mut m_new = m_old;
                #[unroll]
                for c in 0..chunk {
                    if s0 + c * teams + team < bound {
                        m_new = max(m_new, scores[g * chunk + c]);
                    }
                }
                let correction = (m_old - m_new).exp();
                let mut sum = EA::from_int(0);
                #[unroll]
                for c in 0..chunk {
                    let mut weight = EA::from_int(0);
                    if s0 + c * teams + team < bound {
                        weight = (scores[g * chunk + c] - m_new).exp();
                    }
                    weights[g * chunk + c] = weight;
                    sum += weight;
                }
                self.state.m[g] = m_new;
                self.state.l[g] = self.state.l[g] * correction + sum;
                corrections[g] = correction;
            }

            #[unroll]
            for p in 0..per_lane {
                let li = first + p * span;
                if li < lines {
                    #[unroll]
                    for g in 0..rows {
                        let mut acc =
                            self.acc[g * per_lane + p] * Vector::cast_from(corrections[g]);
                        #[unroll]
                        for c in 0..chunk {
                            acc += Vector::<EA, N>::cast_from(vs[c * per_lane + p])
                                * Vector::cast_from(weights[g * chunk + c]);
                        }
                        self.acc[g * per_lane + p] = acc;
                    }
                }
            }
            s0 += comptime!(chunk * teams);
        }
    }

    /// Close the plane's teams into one state: every lane leaves holding the
    /// plane's `(m, l)` and, for its lines, the plane's accumulator — the
    /// split-plane merge, run across lane bits instead of shared memory.
    fn close_teams(&mut self) {
        let teams = comptime!(self.lanes / self.span);
        if comptime!(teams > 1) {
            let rows = comptime!(self.rows);
            let per_lane = comptime!(self.per_lane);
            let team_bits = comptime!((self.lanes - 1) & !(self.span - 1));
            let size!(W1) = 1usize;
            #[unroll]
            for g in 0..rows {
                let m = self.state.m[g];
                let m_all = plane::group::<EA, W1>(Vector::cast_from(m), team_bits, Monoid::Max)
                    .extract(0usize);
                let weight = (m - m_all).exp();
                let l_all = plane::group::<EA, W1>(
                    Vector::cast_from(self.state.l[g] * weight),
                    team_bits,
                    Monoid::Sum,
                )
                .extract(0usize);
                self.state.m[g] = m_all;
                self.state.l[g] = l_all;
                #[unroll]
                for p in 0..per_lane {
                    self.acc[g * per_lane + p] = plane::group::<EA, N>(
                        self.acc[g * per_lane + p] * Vector::cast_from(weight),
                        team_bits,
                        Monoid::Sum,
                    );
                }
            }
        }
    }

    /// The fused ending: the teams close, every plane parks its state and
    /// accumulator lines in shared memory, one `sync_cube`, then plane 0
    /// merges the states across splits and writes the normalized output,
    /// vectorized and cast to the output element. Fully-masked rows store
    /// exact zeros.
    pub fn store<EI: Numeric>(&mut self, out: &mut Tile<EI>, #[comptime] splits: usize) {
        self.close_teams();
        let rows = comptime!(self.rows);
        let per_lane = comptime!(self.per_lane);
        let span = comptime!(self.span);
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
            Shared::<[Vector<EA, N>]>::new_slice(comptime!(rows * splits * per_lane * span));
        let mut maxes = Shared::<[EA]>::new_slice(comptime!(rows * splits));
        let mut sums = Shared::<[EA]>::new_slice(comptime!(rows * splits));
        let lane = UNIT_POS_X as usize;
        let split = UNIT_POS_Y as usize;
        // After the close every team holds the same values; the first one
        // speaks for the plane.
        let speaks = lane < span;
        if lane == 0 {
            #[unroll]
            for g in 0..rows {
                maxes[g * splits + split] = self.state.m[g];
                sums[g * splits + split] = self.state.l[g];
            }
        }
        if speaks {
            #[unroll]
            for g in 0..rows {
                #[unroll]
                for i in 0..per_lane {
                    partials[((g * splits + split) * per_lane + i) * span + lane] =
                        self.acc[g * per_lane + i];
                }
            }
        }
        sync_cube();

        // Plane 0 owns the merge and the store; the other planes are done.
        if split == 0 && speaks {
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
                    let li = lane + i * span;
                    if li < lines {
                        let mut total = Vector::<EA, N>::cast_from(0u32);
                        for t in 0..splits {
                            total += partials[((g * splits + t) * per_lane + i) * span + lane]
                                * Vector::cast_from(weights[t]);
                        }
                        of[g * lines + li] = Vector::cast_from(total * Vector::cast_from(recip));
                    }
                }
            }
        }
    }

    /// The split ending: close the teams, then publish the plane's running
    /// state (lane 0 writes, the states are plane-uniform) and its
    /// accumulator lines into the team's windows of the split-wide buffers.
    /// The caller syncs, merges ([`Tile::merge_splits`]) and drains with the
    /// weights folded in.
    pub fn publish(&mut self, m_win: &mut Tile<EA>, l_win: &mut Tile<EA>, acc_win: &mut Tile<EA>) {
        self.close_teams();
        let rows = comptime!(self.rows);
        // The fold's own share: one worker owning every row, which is what puts
        // lane 0 in range and every other lane past it.
        let share = comptime!(self.state.share);
        m_win.store_rows(&self.state.m, share);
        l_win.store_rows(&self.state.l, share);

        let d = comptime!(self.lines * self.width);
        comptime!(assert!(
            acc_win.space.tile_size() == rows * d,
            "StreamFold::publish: the accumulator window is {{rows, head_dim}}"
        ));
        let per_lane = comptime!(self.per_lane);
        let span = comptime!(self.span);
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
        if lane < span {
            #[unroll]
            for g in 0..rows {
                #[unroll]
                for p in 0..per_lane {
                    let li = lane + p * span;
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
}

/// Close `value` across one team's `span` lanes, leaving each of them holding
/// the team's total. A span of the whole plane is the plane reduction, which
/// also serves a plane width the butterfly cannot split.
#[cube]
fn team_fold<E: Float>(value: E, #[comptime] span: usize, #[comptime] monoid: Monoid) -> E {
    if comptime!(span.is_power_of_two()) {
        let size!(W1) = 1usize;
        plane::group::<E, W1>(Vector::cast_from(value), comptime!(span - 1), monoid).extract(0usize)
    } else {
        plane::reduce::<E>(value, span, monoid)
    }
}
