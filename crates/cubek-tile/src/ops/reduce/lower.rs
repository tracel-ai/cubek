//! Lowering `c.reduce_axis(input, fold)`: at a final tile, the register nest
//! ([`instruction::registers::reduce`](crate::instruction::registers::reduce)); while levels remain,
//! walk this level under its [`Buffering`]. One walk serves every level: what the input costs is
//! its own [`Residence`], and an input that stays put rides a ring of slots that allocate nothing.

use cubecl::prelude::*;

use crate::{instruction::registers::reduce, *};

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = fold(input)`: reduce `input` into `self` across the contracted axes. `self` is a
    /// result, so nothing it held before takes part.
    ///
    /// [`mm`](Tile::mm)'s twin, and the same bargain: where the leaf owns each output cell
    /// outright it starts from `fold`'s identity and never reads `self` back, and where it does
    /// not, the seeding the caller would have written ([`init_identity`](Tile::init_identity))
    /// happens here instead.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>, #[comptime] fold: LeafOp) {
        let spans = comptime!(input.space.spans_contracted_at_leaf(&self.space));
        let overwrites = self.request_overwrite(comptime!(spans));
        if comptime!(!overwrites) {
            self.init_identity(fold);
        }
        lower_reduce(self, input, fold);
        self.request_overwrite(comptime!(false));
    }

    /// `c = fold(c, input)`: [`reduce_axis`](Tile::reduce_axis) with the accumulate `mma` carries
    /// over `mm`, folding each contracted cell into whatever `self` already holds there.
    ///
    /// That existing value is the fold's literal starting point, and the caller owns it: it must
    /// have seeded `self` with `fold`'s identity ([`init_identity`](Tile::init_identity)) before
    /// the first call, or an uninitialized accumulator folds against garbage.
    pub fn reduce_axis_accumulate<In: Numeric>(
        &mut self,
        input: &Tile<In>,
        #[comptime] fold: LeafOp,
    ) {
        lower_reduce(self, input, fold);
    }

    /// The level's operation space: the input operand's space, sized by whichever operand
    /// witnesses each dynamic axis.
    fn reduce_op_space<In: Numeric>(&self, input: &Tile<In>) -> Space {
        let merged = comptime!({
            let merged = input.space.clone();
            assert!(
                self.space.axes().all(|axis| merged.contains(axis)),
                "Tile::reduce_axis: the output spans an axis the input does not, \
                 so the walk would never step it and every region would write the same slice"
            );
            merged
        });
        witnessed_space(merged, self, input, input)
    }
}

/// Lower one operation-scoped accumulator handle to its leaf, and the recursion the walk re-enters
/// per region. Deliberately neither [`reduce_axis`](Tile::reduce_axis) nor
/// [`reduce_axis_accumulate`](Tile::reduce_axis_accumulate): the overwrite is decided once, at the
/// top, from the undivided input space. A region's input is one contracted step of the whole, and
/// always spans its own leaf, so re-deciding down here would overwrite at every step of a walk
/// that must fold them together.
#[cube]
pub(super) fn lower_reduce<Acc: Numeric, In: Numeric>(
    acc: &mut Tile<Acc>,
    input: &Tile<In>,
    #[comptime] fold: LeafOp,
) {
    let partitioner = comptime!(acc.space.partitioner().clone());
    match comptime!(partitioner) {
        Partitioner::Final => reduce_leaf(acc, input, fold),
        Partitioner::Level(level) => {
            let op_space = acc.reduce_op_space(input);
            acc.reduce_buffered(input, fold, op_space, comptime!(level.buffering().depth()));
        }
    }
}

/// Dispatches to the register nest by the accumulator's storage at `Partitioner::Final`.
#[cube]
pub fn reduce_leaf<Acc: Numeric, In: Numeric>(
    acc: &mut Tile<Acc>,
    input: &Tile<In>,
    #[comptime] fold: LeafOp,
) {
    let input_space = comptime!(input.space.clone());
    let vector_size = input.vector_size();
    comptime!(assert!(
        input_space
            .extent_at(input_space.rank() - 1)
            .is_multiple_of(vector_size),
        "reduce: the input's innermost extent must be divisible by its vector size"
    ));

    let space = comptime!(acc.space.clone());
    match &mut acc.tile_kind {
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            reduce::memory(g, input, space, fold);
        }
        TileKind::PlaneTile(t) => {
            reduce_plane_tile(t, input, space, fold);
        }
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "reduce_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            reduce_plane_tile(&mut t, input, space, fold);
        }
        TileKind::TmaGmem(_) => panic!("reduce: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) => panic!("reduce: a procedural tile is not an accumulator sink"),
    }
}

#[cube]
fn reduce_plane_tile<Acc: Numeric, In: Numeric>(
    tile: &mut PlaneTile<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] fold: LeafOp,
) {
    match tile {
        PlaneTile::Register(d) => {
            reduce::register_data(d, input, acc_space, fold);
        }
        PlaneTile::Cmma(_) | PlaneTile::Mma(_) => {
            panic!(
                "reduce: a hardware mma fragment scatters its rows across lanes in a \
                 layout the elementwise walk cannot address; reduce into a register, \
                 Gmem or Smem accumulator instead"
            );
        }
    }
}
