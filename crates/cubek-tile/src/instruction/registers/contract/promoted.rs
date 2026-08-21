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
use crate::*;

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` over the block, one rank-1 update per scalar `K` step.
    ///
    /// The same contraction the memory-backed instruction runs, minus the round trip: there the
    /// block is seeded from the sink and committed back on every visit, so a `K` walk that
    /// returns here repeatedly loses precision to the sink's element between visits. This one
    /// *is* the accumulator, so the partials stay in `T` until [`store_cast_window`] drains them.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(&mut self, lhs: &Tile<EL>, rhs: &Tile<ER>) {
        let pack_l = lhs.quant_pack();
        let pack_r = rhs.quant_pack();
        let vw = rhs.vector_size();
        let lw = lhs.vector_size();
        comptime!(assert!(
            pack_l == 0 && pack_r == 0,
            "RegisterData::mma: a quantized operand against a promoted accumulator is not wired \
             yet — the memory-backed leaf serves those (it dequantizes per read)"
        ));
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma: the block's lines must match the rhs's"
        ));

        let size!(L) = lw;
        let kc = comptime!(rhs.space.extent_at(rhs.space.rank() - 2));
        let (mr, nr) = comptime!((self.mr, self.nr));

        let lhs_mat = lhs.matrix_transparent::<EL, L, L>(0usize);
        // The rhs and the block share the width `RA` (asserted above, `vw == self.vector_size`).
        let rhs_mat = rhs.matrix_transparent::<ER, RA, RA>(0usize);

        let config = comptime!(self.config);
        let unroll = comptime!(mr * nr * vw <= config.budget);
        let lane_fanout = comptime!(config.lane_fanout);

        block::contract::<T, EL, L, ER, RA>(
            &lhs_mat,
            &rhs_mat,
            &mut self.data,
            lw,
            mr,
            nr,
            kc,
            unroll,
            lane_fanout,
        );
    }
}
