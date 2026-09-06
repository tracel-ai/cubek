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

use crate::instruction::registers::contract;
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
        let vector_size = self.vector_size();
        self.accumulator_in::<EA, EL>(lhs, comptime!(PlaneForm::Cmma), vector_size, 1usize, monoid)
    }

    /// [`cmma_accumulator`](Tile::cmma_accumulator) through the manual-mma instruction, whose
    /// fragment transports are `io`'s.
    pub fn mma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] io: MmaIOConfig,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        self.accumulator_in::<EA, EL>(
            lhs,
            comptime!(PlaneForm::Mma { io }),
            vector_size,
            1usize,
            monoid,
        )
    }

    /// [`cmma_accumulator`](Tile::cmma_accumulator) through the software instruction: a register
    /// block per fragment of the grid, run under `config`.
    ///
    /// The block's lines are the rhs's, which is why it reads both operands where the hardware
    /// forms read the lhs alone. An rhs lined along the accumulator gives lines of neighbouring
    /// cells, as wide as this tile's; one lined along the contraction gives each cell a line of
    /// its own partials, folded on drain, and this tile is then scalar. Either way the sum stays
    /// in `EA` across the walk, so a half-precision output summing a long reduction is served
    /// here whatever axis its weight is stored along.
    pub fn block_accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let lw = lhs.vector_size();
        let rw = rhs.vector_size();
        let aw = self.vector_size();
        let fold = comptime!(contract::contracted_per_step(
            &lhs.space.final_space(),
            &rhs.space.final_space(),
            &self.space.final_space(),
            lw,
            rw,
            aw
        ));
        // A memory leaf spreads a line wider than its scalar sink across cells; a block that
        // outlives the leaf has no such step, so its lines and the sink's agree or fold.
        comptime!(assert!(
            fold > 1 || rw == aw,
            "Tile::block_accumulator: the block's lines are the rhs's ({rw} wide) and drain into \
             {aw}-wide cells; a stage served wider than its sink is the memory-backed leaf's \
             (Tile::mma_with)"
        ));
        self.accumulator_in::<EA, EL>(
            lhs,
            comptime!(PlaneForm::Registers { config }),
            rw,
            fold,
            monoid,
        )
    }

    /// [`block_accumulator`](Tile::block_accumulator) for a reduction, whose one operand is
    /// `input`: the block's lines are this tile's, there being no rhs to line them by.
    pub fn block_reducer<EA: Numeric, In: Numeric>(
        &self,
        input: &Tile<In>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        self.accumulator_in::<EA, In>(
            input,
            comptime!(PlaneForm::Registers { config }),
            vector_size,
            1usize,
            monoid,
        )
    }

    /// The plane-resident partition an accumulator contracts in, in `form`, uninitialized and
    /// shaped to meet `lhs` at the instruction. `vector_size` is its lines' width and `fold` what
    /// a line holds ([`RegisterData::fold`]); only the software form reads them.
    pub(crate) fn accumulator_in<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] form: PlaneForm,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let k = comptime!(lhs.space.final_space().contracted_extent(&self.space));
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
            fold,
            lanes,
            monoid,
        )
    }
}
