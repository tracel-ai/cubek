//! The plane-resident accumulators a kernel opens before the walk it spans and drains after
//! ([`drain_cast_into`](Tile::drain_cast_into)): a partition of fragments mirroring the
//! output tile's grid, in the form the leaf contracts through.
//!
//! ```ignore
//! let mut acc = c.cmma_accumulator::<EA, _>(&a, Monoid::Sum);
//! acc.zero();
//! for region in walk { acc.mma(&a.at(&region), &b.at(&region), Semiring::SUM_PROD); }
//! c.drain_cast_into(&acc);
//! ```

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The plane-resident accumulator this output contracts in through the tensor-core
    /// instruction: a partition of cmma fragments mirroring this tile's grid, uninitialized. The
    /// kernel opens it before the walk it spans and drains it after
    /// ([`drain_cast_into`](Tile::drain_cast_into)). `lhs` sizes the contraction depth.
    pub fn cmma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        self.accumulator_in::<EA, EL>(lhs, comptime!(PlaneForm::Cmma), monoid)
    }

    /// [`cmma_accumulator`](Tile::cmma_accumulator) through the manual-mma instruction, whose
    /// fragment transports are `io`'s.
    pub fn mma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] io: MmaIOConfig,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        self.accumulator_in::<EA, EL>(lhs, comptime!(PlaneForm::Mma { io }), monoid)
    }

    /// [`cmma_accumulator`](Tile::cmma_accumulator) through the software instruction: a register
    /// block per fragment of the grid, run under `config`.
    pub fn block_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        self.accumulator_in::<EA, EL>(lhs, comptime!(PlaneForm::Registers { config }), monoid)
    }

    /// The plane-resident partition an accumulator contracts in, in `form`, uninitialized and
    /// shaped to meet `lhs` at the instruction.
    pub(crate) fn accumulator_in<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] form: PlaneForm,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let k = comptime!(lhs.space.final_space().contracted_extent(&self.space));
        let vector_size = self.vector_size();
        let lanes = comptime!(self.space.lanes());
        PlanePartition::<EA>::mirror(
            comptime!(self.space.clone()),
            comptime!(MatrixAxes::accumulator(
                &self.space.final_space(),
                &lhs.space.final_space()
            )),
            comptime!(form),
            comptime!(k),
            vector_size,
            lanes,
            monoid,
        )
    }
}
