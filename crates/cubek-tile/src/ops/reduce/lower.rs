//! Lowering `c.reduce_axis(input, inst)`: at a final tile, the register nest
//! ([`microkernel::reduce`](crate::microkernel::reduce)); while levels remain,
//! walk this level under its [`Buffering`]. One walk serves every level: what the input costs is
//! its own [`Residence`], and an input that stays put rides a ring of slots that allocate nothing.

use cubecl::prelude::*;

use crate::{microkernel::reduce, *};

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.reduce_axis(input, inst)`: reduce `input` into `self` across contracted axes, folding
    /// each contracted cell into whatever `self` already holds there.
    ///
    /// Under `LaneShare::Whole` (a register or memory accumulator, not a lane-shared plane
    /// fragment) that existing value is the fold's literal starting point: nothing seeds it on
    /// `self`'s behalf. The caller must pre-seed `self` with `inst`'s identity ([`Tile::zero`] for
    /// `Sum`, [`Tile::init`] with `E::min_value()`/`E::max_value()` for `Max`/`Min`) before the
    /// first call, or an uninitialized/stale accumulator folds against garbage.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>, #[comptime] inst: LeafOp) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => reduce_leaf(self, input, inst),
            Partitioner::Level(level) => {
                let op_space = self.reduce_op_space(input);
                self.reduce_buffered(input, inst, op_space, comptime!(level.buffering().depth()));
            }
        }
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

/// Dispatches to the register nest by the accumulator's storage at `Partitioner::Final`.
#[cube]
pub fn reduce_leaf<Acc: Numeric, In: Numeric>(
    acc: &mut Tile<Acc>,
    input: &Tile<In>,
    #[comptime] inst: LeafOp,
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
            reduce::memory(g, input, space, inst);
        }
        TileKind::PlaneTile(t) => {
            reduce_plane_tile(t, input, space, inst);
        }
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "reduce_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            reduce_plane_tile(&mut t, input, space, inst);
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
    #[comptime] inst: LeafOp,
) {
    match tile {
        PlaneTile::Register(d) => {
            reduce::register_data(d, input, acc_space, inst);
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
