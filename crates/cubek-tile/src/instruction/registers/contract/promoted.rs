//! The promoted accumulator form of the contraction nest: `acc += lhs · rhs` where the `mr × nr`
//! block *is* the accumulator.
//!
//! The peer of [`memory`](super::memory), and the reason [`block`](super::super::block) takes the
//! block as a parameter. The memory form seeds a block from its sink and commits it back on every
//! visit, so a `K` walk that returns to the leaf repeatedly round-trips its partials through the
//! sink's element; this one keeps them in `T` across the whole walk and only meets memory on
//! drain. Sibling of the two hardware leaves in `instruction/mma`, reached from the same dispatch.

use cubecl::prelude::*;

use super::scale::{ContractEdges, EdgeOrdinal, ScaleLevel, ScaleSide};
use crate::instruction::registers::block;
use crate::instruction::registers::contract::scale_side;
use crate::instruction::registers::lines::{CombinedScales, ScaledLines};
use crate::*;

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` over the block, one rank-1 update per scalar `K` step.
    ///
    /// The same contraction the memory-backed instruction runs, minus the round trip: there the
    /// block is seeded from the sink and committed back on every visit, so a `K` walk that
    /// returns here repeatedly loses precision to the sink's element between visits. This one
    /// *is* the accumulator, so the partials stay in `T` until [`store_cast_window`] drains them.
    ///
    /// A **packed lhs is served here too**: the decode is [`Tile::matrix_packed`]'s and it
    /// happens per read for whichever leaf asks, so a promoted accumulator needs nothing of its
    /// own to serve one. `register_matmul_promoted_accumulator_quant` checks a packed lhs against
    /// a host reference built from the quantized values and their scales.
    ///
    /// The rhs may be packed too, and the width assert below is the whole of what governs it. A
    /// packed operand's `vector_size` is the *served* width — the binding's times the packing
    /// factor — and the rhs's width must be the block's, so a packed rhs is served exactly when
    /// the accumulator is declared at that served width. A caller that declares it narrower is
    /// refused there, by an assert that names both widths; nothing packing-specific is left over
    /// to refuse on its own. `a_packed_rhs_drains_from_a_promoted_accumulator` runs the unscaled
    /// form, and `a_packed_decode_gemv_runs_in_this_spelling` the scaled one, which reaches this
    /// same block through [`mma_scaled`](Self::mma_scaled).
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        comptime!(assert!(
            semiring.add() == self.monoid,
            "RegisterData::mma: this block folds its partials under {:?} and drains them that \
             way, so it cannot contract under {semiring:?}",
            self.monoid
        ));
        let vw = rhs.vector_size();
        let lw = lhs.vector_size();
        // Either operand may be packed: the decode is the read's, and this is the only width
        // either one owes. A packed rhs is served at its packing factor, so it is this assert,
        // not a packing test, that asks the accumulator to have been declared that wide.
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma: the block's lines are {} wide but the rhs serves {vw}; a packed \
             rhs serves its packing factor, so declare the accumulator at that width",
            self.vector_size
        ));
        // The block's lanes are neighbouring cells, so the rhs must line along the accumulator.
        // A contracted axis is one both operands span, which is what this rules out.
        comptime!(assert!(
            !lhs.space.contains(rhs.space.axis_at(rhs.space.rank() - 1)),
            "RegisterData::mma: the rhs lines along a contracted axis, which folds into one cell; \
             the memory-backed leaf serves that step"
        ));

        let size!(L) = lw;
        // Every contracted axis multiplied out: a partitioned contraction carries more than one.
        let kc = comptime!(Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out));
        let (mr, nr) = comptime!((self.mr, self.nr));

        // The accumulator's column edge, which is the rhs's too: read off the operands rather
        // than off the last axis, so a split column group stays one edge.
        let cols = comptime!(MatrixAxes::accumulator(&out, &lhs.space).cols(&out));
        let lhs_axes = comptime!(MatrixAxes::of(&lhs.space, mr, kc));
        let rhs_axes = comptime!(MatrixAxes::of(&rhs.space, kc, cols));
        let lhs_mat = lhs.matrix_packed::<L>(lhs_axes, 0usize);
        // The rhs and the block share the width `RA` (asserted above, `vw == self.vector_size`).
        let rhs_mat = rhs.matrix_packed::<RA>(rhs_axes, 0usize);

        let config = comptime!(self.config);
        let unroll = comptime!(mr * nr * vw <= config.budget);
        let lane_fanout = comptime!(config.lane_fanout);

        block::contract::<T, EL, L, ER, RA, MatrixView<Vector<EL, L>>, MatrixView<Vector<ER, RA>>>(
            &lhs_mat,
            &rhs_mat,
            &mut self.data,
            lw,
            1usize,
            mr,
            nr,
            kc,
            unroll,
            lane_fanout,
            semiring,
        );
    }

    /// `self += (lhs ⊗ scale) · rhs`, or its rhs twin: [`mma`](RegisterData::mma) with one operand
    /// scaled by a real operand, the side read off the scales' axes.
    ///
    /// The form a decode gemv wants: the partials never round-trip through the sink between `K`
    /// steps, so a walk that returns to this leaf keeps them in `T`. `out` is the accumulator's
    /// space, which is what the scales' axes are read against.
    pub(crate) fn mma_scaled<EL: Numeric, ER: Numeric, ES: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        scales: &Sequence<Tile<ES>>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        comptime!(assert!(
            semiring.add() == self.monoid,
            "RegisterData::mma_scaled: this block folds its partials under {:?} and drains them              that way, so it cannot contract under {semiring:?}",
            self.monoid
        ));
        let vw = rhs.vector_size();
        let lw = lhs.vector_size();
        let inner = scales.index(0);
        let sw = inner.vector_size();
        // As [`mma`](Self::mma): a packed rhs — the decode gemv's whole shape — serves its
        // packing factor, so this is the assert that asks the accumulator to be that wide.
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma_scaled: the block's lines are {} wide but the rhs serves {vw}; a \
             packed rhs serves its packing factor, so declare the accumulator at that width",
            self.vector_size
        ));
        // The block's lanes are neighbouring cells, so the rhs must line along the accumulator.
        comptime!(assert!(
            !lhs.space.contains(rhs.space.axis_at(rhs.space.rank() - 1)),
            "RegisterData::mma_scaled: the rhs lines along a contracted axis, which folds into              one cell; the memory-backed leaf serves that step"
        ));

        let size!(L) = lw;
        let size!(S) = sw;
        let kc = comptime!(Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out));
        let (mr, nr) = comptime!((self.mr, self.nr));

        let acc_axes = comptime!(MatrixAxes::accumulator(&out, &lhs.space));
        let cols = comptime!(acc_axes.cols(&out));
        let side = comptime!(scale_side(&inner.space, &out, acc_axes));
        let lhs_axes = comptime!(MatrixAxes::of(&lhs.space, mr, kc));
        let rhs_axes = comptime!(MatrixAxes::of(&rhs.space, kc, cols));
        // This block's own geometry. It walks one contracted value a step, so its accumulator
        // width is the width whichever edge the scales share is served at.
        let operands = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
        let invariant = inner.invariant_over(comptime!(operands.clone()));
        let level = comptime!(ScaleLevel::of(
            &inner.space,
            &ContractEdges {
                mr,
                kc,
                cols,
                reduce: operands
                    .contracting(&out)
                    .iter()
                    .map(|&axis| (axis, operands.extent(axis)))
                    .collect::<Vec<_>>(),
                columns: (acc_axes.col_split..out.rank())
                    .map(|p| (out.axis_at(p), out.extent_at(p)))
                    .collect::<Vec<_>>(),
                lw,
                aw: vw,
                contracted_per_step: 1,
                // This block walks its columns under a constant ordinal, and its rows are the
                // contraction, whose step is a runtime index.
                ordinal: match side {
                    ScaleSide::Rhs => EdgeOrdinal::Constant,
                    ScaleSide::Lhs => EdgeOrdinal::Runtime(
                        "this block walks the contraction at runtime".to_string(),
                    ),
                },
            },
            side,
            &invariant,
            sw,
        ));
        let config = comptime!(self.config);
        let unroll = comptime!(mr * nr * vw <= config.budget);
        let lane_fanout = comptime!(config.lane_fanout);

        // The scale folds into the operand that carries it, so the block contracts one scaled line
        // source against one plain one. Which operand that is decides two types, so two calls.
        match comptime!(side) {
            ScaleSide::Lhs => {
                let lhs_mat = ScaledLines::<MatrixView<Vector<EL, L>>, CombinedScales<ES, S>>::new(
                    lhs.matrix_packed::<L>(lhs_axes, 0usize),
                    super::direct::combined_scales::<ES, S>(scales, comptime!(level), 0usize),
                    comptime!(level.lines_per_scale),
                    comptime!(level.lanes),
                );
                let rhs_mat = rhs.matrix_packed::<RA>(rhs_axes, 0usize);
                block::contract::<
                    T,
                    EL,
                    L,
                    ER,
                    RA,
                    ScaledLines<MatrixView<Vector<EL, L>>, CombinedScales<ES, S>>,
                    MatrixView<Vector<ER, RA>>,
                >(
                    &lhs_mat,
                    &rhs_mat,
                    &mut self.data,
                    lw,
                    1usize,
                    mr,
                    nr,
                    kc,
                    unroll,
                    lane_fanout,
                    semiring,
                );
            }
            ScaleSide::Rhs => {
                let lhs_mat = lhs.matrix_packed::<L>(lhs_axes, 0usize);
                let rhs_mat = ScaledLines::<MatrixView<Vector<ER, RA>>, CombinedScales<ES, S>>::new(
                    rhs.matrix_packed::<RA>(rhs_axes, 0usize),
                    super::direct::combined_scales::<ES, S>(scales, comptime!(level), 0usize),
                    comptime!(level.lines_per_scale),
                    comptime!(level.lanes),
                );
                block::contract::<
                    T,
                    EL,
                    L,
                    ER,
                    RA,
                    MatrixView<Vector<EL, L>>,
                    ScaledLines<MatrixView<Vector<ER, RA>>, CombinedScales<ES, S>>,
                >(
                    &lhs_mat,
                    &rhs_mat,
                    &mut self.data,
                    lw,
                    1usize,
                    mr,
                    nr,
                    kc,
                    unroll,
                    lane_fanout,
                    semiring,
                );
            }
        }
    }
}
