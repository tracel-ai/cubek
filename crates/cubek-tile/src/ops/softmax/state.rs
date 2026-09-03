//! The fold's running state and the masking probe.

use cubecl::prelude::*;

use crate::*;

/// Logits at or below this are treated as masked (effectively -inf). Fits f16.
pub const LOGIT_MASKED: f32 = -6e4;

/// Below this an `l` row sum is numerically zero (fully-masked row).
pub const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;

/// `1/l`, exactly zero when `l` is numerically zero, so a fully-masked row
/// drains to exact zeros instead of NaN.
#[cube]
pub(crate) fn masked_recip<E: Float>(l: E) -> E {
    let eps = E::new(FULLY_MASKED_ROW_THRESHOLD);
    E::cast_from(l >= eps) * clamp_min(l, eps).recip()
}

/// How the score rows are shared out, and therefore how a row reduction closes.
///
/// The two arms compute the same softmax over the same cells; what differs is
/// the worker. Stated once, here, because every op of the leaf has to agree
/// with every other about who owns row `r` — and because the caller's own row
/// loops ([`store_rows`](crate::Tile::store_rows)) have to agree with them too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowShare {
    /// One **unit** per row-slice: the reduction runs in that unit's own
    /// registers over the whole row. No shuffles, no syncs, nothing asked of
    /// the hardware — the arm a device with no plane ops still runs.
    Unit { rows: usize },
    /// One **plane** per row-slice: its lanes split the reduced axis and meet
    /// in a plane reduction, so every lane leaves holding the row's state.
    ///
    /// Costs `lanes`× the workers on the same rows, which is the whole point:
    /// a score tile of 8 rows keeps 8 units busy under `Unit` and a whole
    /// 64-unit cube busy under `Plane`. In exchange the cube's x dim must be a
    /// whole number of planes, and `lanes` must be the width the device
    /// actually commits to — a plane reduction over a wrong width is silently
    /// wrong rather than an error.
    Plane { rows: usize, lanes: usize },
}

impl RowShare {
    /// Rows one worker owns.
    pub fn rows(&self) -> usize {
        match self {
            RowShare::Unit { rows } | RowShare::Plane { rows, .. } => *rows,
        }
    }

    /// Units one worker spans: one, or the plane's width.
    pub fn lanes(&self) -> usize {
        match self {
            RowShare::Unit { .. } => 1,
            RowShare::Plane { lanes, .. } => *lanes,
        }
    }
}

/// Per-row running state `(m, l)` of the online softmax, in the owning
/// worker's registers. Its space is the softmax's kept axes; the score axis it
/// omits is the reduced one. Allocated once before the walk, threaded through
/// every [`Tile::softmax`](crate::Tile::softmax) call (or every
/// [`absorb`](RowState::absorb) of a streamed fold), drained by the epilogue.
#[derive(CubeType)]
pub struct RowState<E: Float> {
    pub m: Array<E>,
    pub l: Array<E>,
    #[cube(comptime)]
    pub space: Space,
    /// Who owns which rows. Under [`RowShare::Plane`] every lane of a plane
    /// holds the same `(m, l)`, since a plane-reduced score is plane-uniform.
    #[cube(comptime)]
    pub share: RowShare,
}

/// What one streamed [`absorb`](RowState::absorb) tells the row's
/// accumulators: rescale what you hold by `correction`, weight the incoming
/// value row by `weight`.
#[derive(CubeType)]
pub struct Rescale<E: Float> {
    /// Rescales the accumulated mix: `exp(m_old - m_new)`.
    pub correction: E,
    /// Weights the new position's value: `exp(score - m_new)`.
    pub weight: E,
}

#[cube]
impl<E: Float> RowState<E> {
    /// `space` is the kept axes; `units` the number of units sharing the
    /// tile, unit u owning rows `[u*rpu, (u+1)*rpu)`.
    pub fn new(#[comptime] space: Space, #[comptime] units: usize) -> RowState<E> {
        let rows = comptime!(space.tile_size().div_ceil(units));
        RowState::<E>::of(space, comptime!(RowShare::Unit { rows }))
    }

    /// [`new`](RowState::new) at plane ownership: `units` units of `lanes`
    /// each, so `units / lanes` planes share the tile and plane `p` owns rows
    /// `[p*rpp, (p+1)*rpp)`, its lanes splitting each row's reduced axis.
    ///
    /// `lanes` must be the width the device commits to
    /// (`plane_size_min == plane_size_max`, and plane ops offered); one lane is
    /// the degenerate case and gives back [`new`](RowState::new)'s arm, which is
    /// what a CPU runtime gets.
    pub fn over_planes(
        #[comptime] space: Space,
        #[comptime] units: usize,
        #[comptime] lanes: usize,
    ) -> RowState<E> {
        let planes = comptime!(units.div_ceil(lanes));
        let rows = comptime!(space.tile_size().div_ceil(planes));
        RowState::<E>::of(space, comptime!(RowShare::Plane { rows, lanes }))
    }

    /// The state one worker holds, at whatever [`RowShare`] the caller states.
    pub fn of(#[comptime] space: Space, #[comptime] share: RowShare) -> RowState<E> {
        let rows = comptime!(share.rows());
        let mut m = Array::new(rows);
        let mut l = Array::new(rows);
        for i in 0..rows {
            m[i] = E::min_value();
            l[i] = E::from_int(0);
        }
        RowState::<E> { m, l, space, share }
    }

    /// Absorb one block's row maxes and sums: `m = max_buf`,
    /// `l = corr*l + sum_buf`. Returns `corr = exp(m_old - m_new)` per row,
    /// the caller's accumulator rescale factor.
    pub fn update(&mut self, max_buf: &Array<E>, sum_buf: &Array<E>) -> Array<E> {
        let rows = comptime!(self.share.rows());
        let mut corr = Array::new(rows);
        for i in 0..rows {
            corr[i] = (self.m[i] - max_buf[i]).exp();
            self.l[i] = corr[i] * self.l[i] + sum_buf[i];
            self.m[i] = max_buf[i];
        }
        corr
    }

    /// Fold one streamed score into row `i`'s `(m, l)`: the per-position
    /// reading of [`update`](RowState::update). The `min_value` identity
    /// makes the first real score overwrite the state cleanly, and a row
    /// that never absorbs keeps `l = 0` for the epilogue's masked guard.
    pub fn absorb(&mut self, i: usize, score: E) -> Rescale<E> {
        let (m_new, l_new, correction, weight) =
            instruction::logsumexp::step::<E>(self.m[i], self.l[i], score);
        self.m[i] = m_new;
        self.l[i] = l_new;
        Rescale::<E> { correction, weight }
    }

    /// Epilogue `lse = m + ln(l)`. Fully-masked rows give -inf via `ln(0)`.
    pub fn lse(&self, i: usize) -> E {
        self.m[i] + E::ln(self.l[i])
    }

    /// Epilogue `1/l` for row `i`, with the fully-masked guard.
    pub fn recip_l(&self, i: usize) -> E {
        masked_recip::<E>(self.l[i])
    }
}

/// The masking predicates and where the score tile sits in the global
/// (kept, reduced) space: origin of its top-left element and the valid
/// extents. Causal and materialized are comptime knobs.
///
/// `q_rows` maps a score row to its query position: a GQA score tile stacks
/// the group members over the same query block (group-major), so row `r` sits
/// at query `origin_q + r % q_rows`. A group-free tile sets `q_rows` to its
/// row count, and the modulo is the identity.
#[derive(CubeType)]
pub struct MaskProbe {
    pub origin_q: usize,
    pub origin_s: usize,
    pub bound_q: usize,
    pub bound_s: usize,
    #[cube(comptime)]
    pub q_rows: usize,
    #[cube(comptime)]
    pub causal: bool,
    #[cube(comptime)]
    pub materialized: bool,
}

#[cube]
impl MaskProbe {
    /// Bound, causal, and materialized predicates at global position (q, s).
    /// `mask` is the `{q, s}` boolean tile (nonzero = masked), read direct
    /// from gmem; only touched when materialized.
    pub fn masked(&self, q: usize, s: usize, mask: &Tile<u32>) -> bool {
        let mut masked = q >= self.bound_q || s >= self.bound_s;
        if comptime!(self.causal) {
            masked = masked || s > q;
        }
        if comptime!(self.materialized) {
            let size!(W) = mask.vector_size();
            let rank = comptime!(mask.space.rank());
            let cols = mask.runtime_extent(comptime!(mask.space.axis_at(rank - 1)));
            masked = masked || mask.flat::<W>().read(q * cols + s).extract(0usize) != 0;
        }
        masked
    }

    /// The query position of score row `r` (see `q_rows`).
    pub(crate) fn row_q(&self, r: usize) -> usize {
        let q_rows = comptime!(self.q_rows);
        self.origin_q + r % q_rows
    }

    /// The probe advanced `offset` along the reduced axis: how a walk hands
    /// each region its own origin.
    pub(crate) fn step_s(&self, offset: usize) -> MaskProbe {
        MaskProbe {
            origin_q: self.origin_q,
            origin_s: self.origin_s + offset,
            bound_q: self.bound_q,
            bound_s: self.bound_s,
            q_rows: comptime!(self.q_rows),
            causal: comptime!(self.causal),
            materialized: comptime!(self.materialized),
        }
    }
}
