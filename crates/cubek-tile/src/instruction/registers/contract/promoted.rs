//! The promoted accumulator form of the contraction nest: `acc += lhs · rhs` where the `mr × nr`
//! block *is* the accumulator.
//!
//! The peer of [`memory`](super::memory), and the reason [`block`](super::super::block) takes the
//! block as a parameter. The memory form seeds a block from its sink and commits it back on every
//! visit, so a `K` walk that returns to the leaf repeatedly round-trips its partials through the
//! sink's element; this one keeps them in `T` across the whole walk and only meets memory on
//! drain. Sibling of the two hardware leaves in `instruction/mma`, reached from the same dispatch.

use cubecl::prelude::*;

use crate::instruction::registers::block;
use crate::instruction::registers::contract::{ScaleSide, scale_side};
use crate::*;

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` over the block, one rank-1 update per scalar `K` step.
    ///
    /// The same contraction the memory-backed instruction runs, minus the round trip: there the
    /// block is seeded from the sink and committed back on every visit, so a `K` walk that
    /// returns here repeatedly loses precision to the sink's element between visits. This one
    /// *is* the accumulator, so the partials stay in `T` until [`store_cast_window`] drains them.
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
        let lhs_packing = lhs.packing();
        let rhs_packing = rhs.packing();
        let vw = rhs.vector_size();
        let lw = lhs.vector_size();
        comptime!(assert!(
            lhs_packing == Packing::Plain && rhs_packing == Packing::Plain,
            "RegisterData::mma: a quantized operand against a promoted accumulator is not wired \
             yet; the memory-backed leaf serves those (it dequantizes per read)"
        ));
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma: the block's lines must match the rhs's"
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

        let cols = comptime!(rhs.space.extent_at(rhs.space.rank() - 1));
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
        scales: &Tile<ES>,
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
        let sw = scales.vector_size();
        comptime!(assert!(
            sw == 1,
            "mm_scaled: the scales are read one value at a time, so their operand is scalar (it              is {sw} wide)"
        ));
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma_scaled: the block's lines must match the rhs's"
        ));
        // The block's lanes are neighbouring cells, so the rhs must line along the accumulator.
        comptime!(assert!(
            !lhs.space.contains(rhs.space.axis_at(rhs.space.rank() - 1)),
            "RegisterData::mma_scaled: the rhs lines along a contracted axis, which folds into              one cell; the memory-backed leaf serves that step"
        ));

        let size!(L) = lw;
        let size!(S) = 1usize;
        let kc = comptime!(Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out));
        let (mr, nr) = comptime!((self.mr, self.nr));

        let cols = comptime!(rhs.space.extent_at(rhs.space.rank() - 1));
        let side = comptime!(scale_side(&scales.space, &out));
        let lhs_axes = comptime!(MatrixAxes::of(&lhs.space, mr, kc));
        let rhs_axes = comptime!(MatrixAxes::of(&rhs.space, kc, cols));
        // The scales carry the values' own axes and address only the ones they span, so their
        // matrix is the same face read through their own projection.
        let scales_axes = comptime!(match side {
            ScaleSide::Lhs => MatrixAxes::of(&scales.space, mr, kc),
            ScaleSide::Rhs => MatrixAxes::of(&scales.space, kc, cols),
        });
        // The scale folds in under the operand's view, so the block below runs the plain
        // contraction.
        let lhs_mat = match comptime!(side) {
            ScaleSide::Lhs => lhs.matrix_scaled::<L, ES, S>(lhs_axes, scales, scales_axes, 0),
            ScaleSide::Rhs => lhs.matrix_packed::<L>(lhs_axes, 0usize),
        };
        let rhs_mat = match comptime!(side) {
            ScaleSide::Lhs => rhs.matrix_packed::<RA>(rhs_axes, 0usize),
            ScaleSide::Rhs => rhs.matrix_scaled::<RA, ES, S>(rhs_axes, scales, scales_axes, 0),
        };

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
}
