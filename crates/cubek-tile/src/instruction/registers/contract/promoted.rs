//! The promoted accumulator form of the contraction nest: `acc += lhs · rhs` where the `mr × nr`
//! block *is* the accumulator.
//!
//! The peer of [`memory`](super::memory), and the reason [`block`](super::super::block) takes the
//! block as a parameter. The memory form seeds a block from its sink and commits it back on every
//! visit, so a `K` walk that returns to the leaf repeatedly round-trips its partials through the
//! sink's element; this one keeps them in `T` across the whole walk and only meets memory on
//! drain. Sibling of the two hardware leaves in `instruction/mma`, reached from the same dispatch.

use cubecl::prelude::*;

use super::factor::{level_of, scale_width, scales_of, spread};
use super::scale::{ContractEdges, EdgeOrdinal, Side};
use crate::instruction::registers::block;
use crate::instruction::registers::lines::{CombinedScales, ScaledLines};
use crate::*;

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` over the block, one rank-1 update per scalar `K` step, each factor
    /// times whatever scales it carries.
    ///
    /// The same contraction the memory-backed instruction runs, minus the round trip: there the
    /// block is seeded from the sink and committed back on every visit, so a `K` walk that
    /// returns here repeatedly loses precision to the sink's element between visits. This one
    /// *is* the accumulator, so the partials stay in `T` until [`store_cast_window`] drains them.
    ///
    /// A **packed factor is served here too**: the decode is [`Tile::matrix_packed`]'s and it
    /// happens per read for whichever leaf asks, so a promoted accumulator needs nothing of its
    /// own to serve one. The rhs may be packed as well, and the width assert below is the whole
    /// of what governs it: a packed operand's `vector_size` is the *served* width, and the block
    /// was opened at the rhs's.
    ///
    /// An rhs lined along the contraction is the folded step ([`fold`](Self::fold)): a step
    /// consumes a whole line of each operand and the block's lanes are one cell's partials,
    /// which [`store_cast_window`](Self::store_cast_window) collapses. The block must have been
    /// opened that way: the lines are allocated at the open, not here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mma<EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
        &mut self,
        lhs: &Scaled<EL, LS>,
        rhs: &Scaled<ER, RS>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        let lhs_values = lhs.values();
        let rhs_values = rhs.values();
        let lhs_levels = lhs.levels();
        let rhs_levels = rhs.levels();
        let lhs_count = lhs_levels.len();
        let rhs_count = rhs_levels.len();
        let scaled = comptime!(lhs_count > 0 || rhs_count > 0);

        comptime!(assert!(
            semiring.add() == self.monoid,
            "RegisterData::mma: this block folds its partials under {:?} and drains them that \
             way, so it cannot contract under {semiring:?}",
            self.monoid
        ));
        let vw = rhs_values.vector_size();
        let lw = lhs_values.vector_size();
        let fold = comptime!(self.fold);
        // Either factor may be packed: the decode is the read's, and this is the only width
        // either one owes. A packed rhs is served at its packing factor, so it is this assert,
        // not a packing test, that asks the block to have been opened by the same rhs.
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma: the block's lines are {} wide but the rhs serves {vw}; a packed \
             rhs serves its packing factor, so open the block against the rhs it contracts",
            self.vector_size
        ));
        // A contracted axis is one both operands span. The rhs lining along one is the folded
        // step, and the block's lines mean one thing for the whole walk.
        let lined_along_k = comptime!(
            lhs_values
                .space
                .contains(rhs_values.space.axis_at(rhs_values.space.rank() - 1))
        );
        comptime!(assert!(
            lined_along_k == (fold > 1),
            "RegisterData::mma: the block's lines hold {fold} partials of a cell but the rhs lines \
             along {}; open the block against the operands it contracts",
            if lined_along_k {
                "the contraction"
            } else {
                "the accumulator"
            }
        ));
        comptime!(assert!(
            fold == 1 || lw == fold,
            "RegisterData::mma: a step consumes {fold} contracted values, so the lhs must line \
             along the contraction at that width (it is {lw} wide)"
        ));
        // A scaled step walks one contracted value at a time, so this block's lanes are
        // neighbouring cells: a folded step here would need the scale level to step by lines.
        comptime!(assert!(
            !scaled || fold == 1,
            "RegisterData::mma: the rhs lines along a contracted axis, which folds into one \
             cell; the memory-backed leaf serves a scaled step of that shape (Tile::mma_with)"
        ));

        let size!(L) = lw;
        let lsw = scale_width(&lhs_levels);
        let rsw = scale_width(&rhs_levels);
        let size!(LSW) = lsw;
        let size!(RSW) = rsw;

        // Every contracted axis multiplied out: a partitioned contraction carries more than one.
        let operands = comptime!(Space::merge(&[&lhs_values.space, &rhs_values.space]));
        let kc = comptime!(operands.contracted_extent(&out));
        let (mr, nr) = comptime!((self.mr, self.nr));

        // The accumulator's column edge, which is the rhs's too: read off the operands rather
        // than off the last axis, so a split column group stays one edge.
        let acc_axes = comptime!(MatrixAxes::accumulator(&out, &lhs_values.space));
        let cols = comptime!(acc_axes.cols(&out));
        let lhs_axes = comptime!(MatrixAxes::of(&lhs_values.space, mr, kc));
        // Lined along the contraction the rhs reads as `(col, k)`, along the accumulator `(k, col)`.
        let rhs_axes = comptime!(if fold > 1 {
            MatrixAxes::of(&rhs_values.space, cols, kc)
        } else {
            MatrixAxes::of(&rhs_values.space, kc, cols)
        });

        // This block's own edges, which is all a scale level needs of it. It walks one contracted
        // value a step, so its accumulator width is the width whichever edge a factor's scales
        // share is served at. Its columns are walked under a constant ordinal and its rows are
        // the contraction, whose step is a runtime index, so the two factors differ in that
        // alone.
        let reduce = comptime!(
            operands
                .contracting(&out)
                .iter()
                .map(|&axis| (axis, operands.extent(axis)))
                .collect::<Vec<_>>()
        );
        let columns = comptime!(
            (acc_axes.col_split..out.rank())
                .map(|p| (out.axis_at(p), out.extent_at(p)))
                .collect::<Vec<_>>()
        );
        let lhs_level = level_of(
            &lhs_levels,
            comptime!(operands.clone()),
            comptime!(out.clone()),
            comptime!(acc_axes),
            comptime!(ContractEdges {
                mr,
                kc,
                cols,
                reduce: reduce.clone(),
                columns: columns.clone(),
                lw,
                aw: vw,
                contracted_per_step: 1,
                ordinal: EdgeOrdinal::Runtime(
                    "this block walks the contraction at runtime".to_string()
                ),
            }),
            comptime!(Side::Lhs),
        );
        let rhs_level = level_of(
            &rhs_levels,
            comptime!(operands.clone()),
            comptime!(out.clone()),
            comptime!(acc_axes),
            comptime!(ContractEdges {
                mr,
                kc,
                cols,
                reduce,
                columns,
                lw,
                aw: vw,
                contracted_per_step: 1,
                ordinal: EdgeOrdinal::Constant,
            }),
            comptime!(Side::Rhs),
        );
        let lhs_spread = comptime!(spread(lhs_level));
        let rhs_spread = comptime!(spread(rhs_level));

        let config = comptime!(self.config);
        let unroll = comptime!(mr * nr * vw <= config.budget);
        let lane_fanout = comptime!(config.lane_fanout);

        // Each factor as the block reads it: its values, times whatever scales it carries.
        let lhs_mat = ScaledLines::<MatrixView<Vector<EL, L>>, CombinedScales<LS, LSW>>::new(
            lhs_values.matrix_packed::<L>(lhs_axes, 0usize),
            scales_of::<LS, LSW>(&lhs_levels, comptime!(lhs_level), 0usize),
            comptime!(lhs_spread.0),
            comptime!(lhs_spread.1),
        );
        // The rhs and the block share the width `RA` (asserted above, `vw == self.vector_size`).
        let rhs_mat = ScaledLines::<MatrixView<Vector<ER, RA>>, CombinedScales<RS, RSW>>::new(
            rhs_values.matrix_packed::<RA>(rhs_axes, 0usize),
            scales_of::<RS, RSW>(&rhs_levels, comptime!(rhs_level), 0usize),
            comptime!(rhs_spread.0),
            comptime!(rhs_spread.1),
        );

        block::contract::<
            T,
            EL,
            L,
            ER,
            RA,
            ScaledLines<MatrixView<Vector<EL, L>>, CombinedScales<LS, LSW>>,
            ScaledLines<MatrixView<Vector<ER, RA>>, CombinedScales<RS, RSW>>,
        >(
            &lhs_mat,
            &rhs_mat,
            &mut self.data,
            lw,
            fold,
            mr,
            nr,
            kc,
            unroll,
            lane_fanout,
            semiring,
        );
    }
}
