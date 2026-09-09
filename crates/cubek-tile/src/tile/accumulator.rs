//! The plane-resident accumulator an output opens: a partition of fragments mirroring the output's
//! grid, contracted into under the levels below the open, and stored back one fragment per cell
//! by the loop the kernel writes over them ([`Tile::copy_cast_from`]).
//!
//! ```ignore
//! let mut acc = c.cmma_accumulator::<EA, EL>(&a, fragments, Monoid::Sum);
//! acc.zero();
//! for step in plane.walk(steps) { /* acc.at(&cell).mma(..) */ }
//! for cell in plane.walk(cells).unrolled() {
//!     c.at(&cell).copy_cast_from(&acc.at(&cell));
//! }
//! ```

use cubecl::prelude::*;

use crate::instruction::registers::contract;
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

    /// [`new`](Fragments::new) under the levels below `out`'s own depth in its partitioning:
    /// what a kernel handed a partitioning reads off the tile it opens the accumulator on, as
    /// `comptime!(Fragments::below(&out, &lhs))`.
    pub fn below(out: &impl Placed, lhs: &impl Placed) -> Self {
        Fragments::new(out.space(), lhs.space(), out.below())
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The plane-resident accumulator this output contracts in through the tensor-core
    /// instruction: a partition of cmma fragments mirroring this tile's grid, uninitialized. The
    /// kernel opens it before the walk it spans and stores it after, one fragment per cell
    /// ([`copy_cast_from`](Tile::copy_cast_from)). `lhs` sizes the contraction depth.
    pub fn cmma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] fragments: Fragments,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        self.accumulator_in::<EA, EL>(
            lhs,
            fragments,
            comptime!(PlaneForm::Cmma),
            vector_size,
            1usize,
            monoid,
        )
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
        let vector_size = self.vector_size();
        self.accumulator_in::<EA, EL>(
            lhs,
            fragments,
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
        #[comptime] fragments: Fragments,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let lw = lhs.vector_size();
        let rw = rhs.vector_size();
        let aw = self.vector_size();
        let fold = comptime!(contract::contracted_per_step(
            &lhs.space,
            &rhs.space,
            &self.space,
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
            fragments,
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
        #[comptime] fragments: Fragments,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        self.accumulator_in::<EA, In>(
            input,
            fragments,
            comptime!(PlaneForm::Registers { config }),
            vector_size,
            1usize,
            monoid,
        )
    }

    /// This plane-resident accumulator opened for row-wise ops: a scratch of one tile in shared
    /// memory per plane, `planes` of them for the cube's planes of `lanes` units, that
    /// [`rescale_rows`](Tile::rescale_rows) bounces each tile through. Stated where the
    /// accumulator opens, since the scratch is part of its residence.
    pub fn with_scratch(self, #[comptime] planes: usize, #[comptime] lanes: usize) -> Tile<Acc> {
        let space = comptime!(self.space.clone());
        let depth = comptime!(self.depth);
        let levels = comptime!(self.levels.clone());
        match self.tile_kind {
            TileKind::PlanePartition(p) => {
                let (m, n) = p.at(0usize, 0usize).shape();
                let cells = comptime!(m * n);
                let start = (UNIT_POS as usize / lanes) * cells;
                let end = start + cells;
                let scratch = Shared::<[Acc]>::new_slice(comptime!(cells * planes))
                    .map(|scratch| &scratch[start..end]);
                // Every fragment carries the scratch too: one taken off the partition by `at`
                // bounces on its own, which is how it drains into a store that folds.
                let mut frags = Sequence::<PlaneTile<Acc>>::new();
                #[unroll]
                for i in 0..comptime!(p.m_tiles * p.n_tiles) {
                    frags.push(
                        p.frags
                            .index(i)
                            .clone()
                            .with_scratch(scratch.clone(), lanes),
                    );
                }
                Tile::<Acc> {
                    tile_kind: TileKind::new_PlanePartition(PlanePartition::<Acc> {
                        frags,
                        m_tiles: comptime!(p.m_tiles),
                        n_tiles: comptime!(p.n_tiles),
                        scratch: ComptimeOption::new_Some(scratch),
                    }),
                    space,
                    depth,
                    levels,
                }
            }
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("Tile::with_scratch: a scratch backs a plane-resident accumulator")
            }
        }
    }

    /// The plane-resident partition an accumulator contracts in, in `form`, uninitialized and
    /// shaped to meet `lhs` at the instruction. `vector_size` is its lines' width and `fold` what
    /// a line holds ([`RegisterData::fold`]); only the software form reads them.
    pub(crate) fn accumulator_in<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] fragments: Fragments,
        #[comptime] form: PlaneForm,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        PlanePartition::<EA>::mirror(
            comptime!(self.space.clone()),
            comptime!(MatrixAxes::accumulator(&self.space, &lhs.space)),
            comptime!(form),
            comptime!(fragments),
            vector_size,
            fold,
            monoid,
            comptime!(self.depth),
            comptime!(self.levels.clone()),
        )
    }
}
