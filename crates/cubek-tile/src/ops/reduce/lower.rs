//! Lowering `c.reduce_axis(input, fold)`: at a final tile, the register nest
//! ([`instruction::registers::reduce`](crate::instruction::registers::reduce)); while levels remain,
//! walk this level under its [`Buffering`]. One walk serves every level: what the input costs is
//! its own [`Residence`], and an input that stays put rides a ring of slots that allocate nothing.

use cubecl::prelude::*;

use crate::{instruction::registers::reduce, *};

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.reduce_axis(input, fold)`: reduce `input` into `self` across contracted axes, folding
    /// each contracted cell into whatever `self` already holds there.
    ///
    /// Where `self` is known to hold `fold`'s identity ([`Tile::init_identity`]) and the
    /// reduction lands whole at the leaf
    /// ([`spans_contracted_at_leaf`](Space::spans_contracted_at_leaf)), the initial sink read is
    /// replaced by that identity instead. Otherwise `self`'s existing value is the fold's literal
    /// starting point: the caller must have seeded it, or an uninitialized/stale accumulator folds
    /// against garbage.
    ///
    /// The stamp is consumed here, so a caller-owned loop folding over its own contraction steps
    /// must seed *inside* the loop or not at all: seeding above it serves the first step and
    /// leaves the rest reading a sink they never wrote.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>, #[comptime] fold: LeafOp) {
        let replaces = self.replaces_reduce_sink(input, fold);
        self.set_sink_identity(comptime!(if replaces { Some(fold) } else { None }));
        lower_reduce(self, input, fold);
        self.set_sink_identity(comptime!(None));
    }

    /// Whether this reduction may seed from the identity instead of reading the sink: the buffer
    /// must be known to hold `fold`'s identity, and the final tile must span every contracted
    /// axis whole, so the walk above the leaf never returns to a cell it has already written.
    fn replaces_reduce_sink<In: Numeric>(
        &self,
        input: &Tile<In>,
        #[comptime] fold: LeafOp,
    ) -> comptime_type!(bool) {
        let sink_id = self.sink_identity();
        comptime!(sink_id == Some(fold) && input.space.spans_contracted_at_leaf(&self.space))
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

/// Lower one operation-scoped accumulator handle to its leaf. `sink_identity` is derived before
/// this recursion from the undivided operand spaces. When set to `Some(fold)`, every contracted axis already
/// fits in the final space, so no ancestor visits an output cell for multiple contracted regions;
/// child handles preserve the stamp unchanged and never recompute it from their smaller spaces.
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
