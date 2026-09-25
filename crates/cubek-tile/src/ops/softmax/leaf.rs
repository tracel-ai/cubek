//! The softmax leaf, composed from the row ops at the legacy `softmax_at`
//! granularity.
//!
//! Row ownership is the state's statement ([`RowShare`]) and this leaf's only branch: a worker
//! owns a contiguous slice of the score tile's rows and keeps their running state in registers.
//!
//! The worker is a unit ([`rowwise`](super::rowwise)) or a plane whose units split the reduced
//! axis ([`planewise`](super::planewise)); neither arm reads another worker's cell, so no syncs.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// One online-softmax fold step on this final score tile, in place: scale and mask, row-max
    /// against the running max, exponentiate, row-sum, cast-write the unnormalized P tile, state
    /// update. Returns the accumulator rescale `exp(m_old - m_new)` per owned row (1 if unowned).
    ///
    /// The caller owns the walk and the epilogue ([`RowState::recip_l`], [`RowState::lse`]). The
    /// reduced axis is the score axis absent from `state`'s space.
    ///
    /// [`softmax_in_place`](Tile::softmax_in_place) is the same step without
    /// the P tile: the exponentiated scores *are* the probabilities, at the
    /// score's own element.
    pub fn softmax<EP: Float>(
        &mut self,
        p: &mut Tile<EP>,
        state: &mut RowState<EA>,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        scale: EA,
    ) -> Array<EA> {
        let corr = self.softmax_in_place(state, probe, mask, scale);
        match comptime!(state.share) {
            RowShare::Unit { rows: _ } => self.write_rows_to(p, &*state),
            RowShare::Plane { rows, units } => self.write_rows_to_planar(p, rows, units),
        }
        corr
    }

    /// [`softmax`](Tile::softmax) with the probabilities left where the scores were: after it,
    /// `self` holds this step's unnormalized P, which the value matmul contracts. One tile, one
    /// pass fewer, no cast: the mix reads P at the accumulate element, as its hardware arm does.
    pub fn softmax_in_place(
        &mut self,
        state: &mut RowState<EA>,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        scale: EA,
    ) -> Array<EA> {
        let rank = comptime!(self.place.space.rank());
        // Rank, not finality: a score tile states the instruction its matmuls contract through,
        // and that statement is a level. Every read here is a flat one over the whole tile, which
        // is the same cells whatever cuts them.
        comptime!(assert!(
            rank == 2,
            "softmax: a leaf op on rank-2 score tiles"
        ));
        comptime!(assert!(
            state.space.contains(self.place.space.axis_at(0))
                && !state.space.contains(self.place.space.axis_at(1)),
            "softmax reduces the score axis absent from the state's space; \
             v1 requires it to be the trailing axis"
        ));
        // The passes read a row a line at a time, so the lines must not straddle rows.
        let w = self.vector_size();
        comptime!(assert!(
            self.place.space.extent_at(1).is_multiple_of(w),
            "softmax: the score's line width divides its columns"
        ));

        let rows = comptime!(state.share.rows());
        let mut max_buf = Array::<EA>::new(rows);
        let mut sum_buf = Array::<EA>::new(rows);

        match comptime!(state.share) {
            RowShare::Unit { rows: _ } => {
                self.scale_and_mask(scale, probe, mask, &*state);
                self.row_max(&mut max_buf, &*state);
                self.exp_diff(&max_buf, &*state);
                self.row_sum(&mut sum_buf, &*state);
            }
            RowShare::Plane { rows, units } => {
                self.scale_and_mask_planar(scale, probe, mask, rows, units);
                self.row_max_planar(&mut max_buf, &state.m, rows, units);
                self.exp_diff_planar(&max_buf, rows, units);
                self.row_sum_planar(&mut sum_buf, rows, units);
            }
        }

        state.update(&max_buf, &sum_buf)
    }
}
