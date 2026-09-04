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

/// The shape a plane-resident accumulator is opened at: the `m_tiles × n_tiles` fragments one
/// plane holds, each `m × n` and contracting `k` a step. Stated where the accumulator opens,
/// before the loops that walk it exist; the walk then checks itself against it (a level's grid
/// must divide the partition, the leaf must be one fragment).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fragments {
    pub m_tiles: usize,
    pub n_tiles: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl Fragments {
    /// The shape an accumulator over `out` contracting `lhs` has under `levels`, the ones below
    /// the site it opens at: what a kernel handed its levels reads off them. A blueprint that
    /// holds the shape states the fields directly.
    pub fn new(out: &Space, lhs: &Space, levels: &[Level]) -> Self {
        let (m_tiles, n_tiles) = partition_shape(out, levels);
        let leaf = out.leaf(levels);
        let lhs_leaf = lhs.leaf(levels);
        let axes = MatrixAxes::accumulator(&leaf, &lhs_leaf);
        // The edges the accumulator's own axes give, not its last two: a split column group is
        // one edge, and sizing the block off the innermost axis alone would cut it in half.
        Fragments {
            m_tiles,
            n_tiles,
            m: axes.rows(&leaf),
            n: axes.cols(&leaf),
            k: lhs_leaf.contracted_extent(out),
        }
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The plane-resident accumulator this output contracts in through the tensor-core
    /// instruction: a partition of cmma fragments mirroring this tile's grid, uninitialized. The
    /// kernel opens it before the walk it spans and drains it after
    /// ([`drain_cast_into`](Tile::drain_cast_into)). `lhs` sizes the contraction depth.
    pub fn cmma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] fragments: Fragments,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        self.accumulator_in::<EA, EL>(lhs, fragments, comptime!(PlaneForm::Cmma), monoid)
    }

    /// [`cmma_accumulator`](Tile::cmma_accumulator) through the manual-mma instruction, whose
    /// fragment transports are `io`'s.
    pub fn mma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] fragments: Fragments,
        #[comptime] io: MmaIOConfig,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        self.accumulator_in::<EA, EL>(lhs, fragments, comptime!(PlaneForm::Mma { io }), monoid)
    }

    /// [`cmma_accumulator`](Tile::cmma_accumulator) through the software instruction: a register
    /// block per fragment of the grid, run under `config`.
    pub fn block_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] fragments: Fragments,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        self.accumulator_in::<EA, EL>(
            lhs,
            fragments,
            comptime!(PlaneForm::Registers { config }),
            monoid,
        )
    }

    /// The plane-resident partition an accumulator contracts in, in `form`, uninitialized and
    /// shaped to meet `lhs` at the instruction.
    pub(crate) fn accumulator_in<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] fragments: Fragments,
        #[comptime] form: PlaneForm,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        PlanePartition::<EA>::mirror(
            comptime!(self.space.clone()),
            comptime!(MatrixAxes::accumulator(&self.space, &lhs.space)),
            comptime!(form),
            comptime!(fragments),
            vector_size,
            monoid,
        )
    }
}
