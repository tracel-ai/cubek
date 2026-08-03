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
pub fn masked_recip<E: Float>(l: E) -> E {
    let eps = E::new(FULLY_MASKED_ROW_THRESHOLD);
    E::cast_from(l >= eps) * clamp_min(l, eps).recip()
}

/// Per-row running state `(m, l)` of the online softmax, in the owning
/// unit's registers. Its space is the softmax's kept axes; the score axis it
/// omits is the reduced one. Allocated once before the walk, threaded through
/// every [`Tile::softmax`](crate::Tile::softmax) call (or every
/// [`absorb`](RowState::absorb) of a streamed fold), drained by the epilogue.
#[derive(CubeType)]
pub struct RowState<E: Float> {
    pub m: Array<E>,
    pub l: Array<E>,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub rows_per_unit: usize,
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
        let rows_per_unit = comptime!(space.tile_size().div_ceil(units));
        let mut m = Array::new(rows_per_unit);
        let mut l = Array::new(rows_per_unit);
        for i in 0..rows_per_unit {
            m[i] = E::min_value();
            l[i] = E::from_int(0);
        }
        RowState::<E> {
            m,
            l,
            space,
            rows_per_unit,
        }
    }

    /// Absorb one block's row maxes and sums: `m = max_buf`,
    /// `l = corr*l + sum_buf`. Returns `corr = exp(m_old - m_new)` per row,
    /// the caller's accumulator rescale factor.
    pub fn update(&mut self, max_buf: &Array<E>, sum_buf: &Array<E>) -> Array<E> {
        let mut corr = Array::new(self.rows_per_unit);
        for i in 0..self.rows_per_unit {
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
        let m_new = max(self.m[i], score);
        let correction = (self.m[i] - m_new).exp();
        let weight = (score - m_new).exp();
        self.l[i] = self.l[i] * correction + weight;
        self.m[i] = m_new;
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
            masked = masked || mask.flat::<W>().read(q * cols + s).extract(0) != 0;
        }
        masked
    }

    /// The query position of score row `r` (see `q_rows`).
    pub fn row_q(&self, r: usize) -> usize {
        let q_rows = comptime!(self.q_rows);
        self.origin_q + r % q_rows
    }

    /// The probe advanced `offset` along the reduced axis: how a walk hands
    /// each region its own origin.
    pub fn step_s(&self, offset: usize) -> MaskProbe {
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
