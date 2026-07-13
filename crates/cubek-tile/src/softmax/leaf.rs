//! The softmax leaf, composed from the [`rowwise`](super::rowwise) ops at the
//! legacy `softmax_at` granularity.
//!
//! Row ownership is dictated, not asked: each unit owns a fixed contiguous
//! slice of the score tile's rows and keeps the running state for those rows
//! in its own registers. No plane shuffles, no fragment-layout knowledge, no
//! syncs (units touch only their own rows). A single unit owning all rows is
//! the unit/CPU path; one row per unit is the plane path.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// One online-softmax fold step on this final score tile, in place:
    /// scale + mask, row-max against the running max, exponentiate, row-sum,
    /// cast-write the (unnormalized) P tile, state update. Returns
    /// `corr = exp(m_old - m_new)` per owned row, the caller's accumulator
    /// rescale factor (1 for unowned slots). The caller owns the walk and the
    /// epilogue ([`RowState::recip_l`], [`RowState::lse`]).
    ///
    /// The reduced axis is the score axis absent from `state`'s space.
    pub fn softmax<EP: Float>(
        &mut self,
        p: &mut Tile<EP>,
        state: &mut RowState<EA>,
        probe: &MaskProbe,
        mask: &Tile<u32>,
        scale: EA,
    ) -> Array<EA> {
        let rank = comptime!(self.space.rank());
        comptime!(assert!(
            self.space.is_final() && rank == 2,
            "softmax: a leaf op on final rank-2 score tiles"
        ));
        comptime!(assert!(
            state.space.contains(self.space.axis_at(0))
                && !state.space.contains(self.space.axis_at(1)),
            "softmax reduces the score axis absent from the state's space; \
             v1 requires it to be the trailing axis"
        ));
        // Row-serial scalar reads; a vectorized leaf is a later drop-in swap.
        let w = self.vector_size();
        let wp = p.vector_size();
        comptime!(assert!(
            w == 1 && wp == 1,
            "softmax: vectorized tiles not supported yet"
        ));

        let rpu = comptime!(state.rows_per_unit);
        let mut max_buf = Array::<EA>::new(rpu);
        let mut sum_buf = Array::<EA>::new(rpu);

        self.scale_and_mask(scale, probe, mask, rpu);
        self.row_max(&mut max_buf, &state.m, rpu);
        self.exp_diff(&max_buf, rpu);
        self.row_sum(&mut sum_buf, rpu);
        self.write_rows_to(p, rpu);

        state.update(&max_buf, &sum_buf)
    }
}
