//! The promoted accumulator form of the contraction nest: `acc += lhs · rhs` where the `mr × nr`
//! block *is* the accumulator.
//!
//! The peer of [`memory`](super::memory), and the reason [`block`](super::super::registers) takes the
//! block as a parameter: the memory form round-trips its partials through the sink's element on
//! every visit, this one keeps them in `T` across the walk and only meets memory on drain.
//!
//! Sibling of the two hardware leaves in `instruction/mma`, reached from the same dispatch.

use cubecl::prelude::*;

use super::registers;
use super::scale::Side;
use crate::*;

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` over the block, one rank-1 update per scalar `K` step, each factor
    /// times whatever scales it carries.
    ///
    /// The memory-backed instruction's contraction minus the round trip: there the block is seeded
    /// from the sink and committed back per visit, losing precision to the sink's element. This one
    /// *is* the accumulator, so partials stay in `T` until [`store_cast_window`] drains them.
    ///
    /// A **packed factor is served here too**: the decode is [`Tile::matrix_packed`]'s, per read
    /// for whichever leaf asks. The rhs may be packed as well; the width assert below governs it,
    /// as a packed `vector_size` is the *served* width, and the block was opened at the rhs's.
    ///
    /// An rhs lined along the contraction is the folded step ([`fold`](Self::fold)): a step
    /// consumes a whole line of each operand and the block's lanes are one cell's partials, which
    /// [`store_cast_window`](Self::store_cast_window) collapses. The block must be opened that way.
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
            lhs_values.place.space.contains(
                rhs_values
                    .place
                    .space
                    .axis_at(rhs_values.place.space.rank() - 1)
            )
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
        let size!(L) = lw;

        // Every contracted axis multiplied out: a partitioned contraction carries more than one.
        let operands = comptime!(Space::merge(&[
            &lhs_values.place.space,
            &rhs_values.place.space
        ]));
        let kc = comptime!(operands.contracted_extent(&out));
        let (mr, nr) = comptime!((self.mr, self.nr));

        // The accumulator's column edge, which is the rhs's too: read off the operands rather
        // than off the last axis, so a split column group stays one edge.
        let acc_axes = comptime!(MatrixAxes::accumulator(&out, &lhs_values.place.space));
        let cols = comptime!(acc_axes.cols(&out));
        let lhs_axes = comptime!(
            MatrixAxes::new(&lhs_values.place.space, mr, kc).unwrap_or_else(|e| panic!("{e}"))
        );
        // Lined along the contraction the rhs reads as `(col, k)`, along the accumulator
        // `(k, col)`.
        let rhs_axes = comptime!(if fold > 1 {
            MatrixAxes::new(&rhs_values.place.space, cols, kc).unwrap_or_else(|e| panic!("{e}"))
        } else {
            MatrixAxes::new(&rhs_values.place.space, kc, cols).unwrap_or_else(|e| panic!("{e}"))
        });

        let config = comptime!(self.config);
        let unroll = comptime!(mr * nr * vw <= config.budget);
        let lane_fanout = comptime!(config.lane_fanout);

        // Each factor as the block reads it: its values' matrix, and its scales looked up at
        // every line's own coordinates. The rhs and the block share the width `RA` (asserted
        // above, `vw == self.vector_size`).
        let lhs_mat = lhs_values.matrix_packed::<L>(lhs_axes, 0usize);
        let rhs_mat = rhs_values.matrix_packed::<RA>(rhs_axes, 0usize);
        let lhs_scales = lhs.lookup(
            lhs_axes,
            0usize,
            comptime!(Side::Lhs),
            comptime!(out.clone()),
            comptime!(acc_axes),
        );
        let rhs_scales = rhs.lookup(
            rhs_axes,
            0usize,
            comptime!(Side::Rhs),
            comptime!(out.clone()),
            comptime!(acc_axes),
        );

        registers::contract::<T, EL, L, LS, ER, RA, RS>(
            &lhs_mat,
            &lhs_scales,
            &rhs_mat,
            &rhs_scales,
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
