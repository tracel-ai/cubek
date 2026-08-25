//! Lowering `c.reduce_axis(input, monoid)` and its accumulating twin: at a final tile, the
//! register nest
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
    /// outright it starts from the monoid's identity and never reads `self` back, and where it
    /// does not, the seeding the caller would have written
    /// ([`init_identity`](Tile::init_identity)) happens here instead.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>, #[comptime] monoid: Monoid) {
        let spans = comptime!(match input.space.spans_contracted_at_leaf(&self.space) {
            true => InitFrom::Identity,
            false => InitFrom::Cell,
        });
        let init_from = self.request_init_from(comptime!(spans));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(monoid),
        }
        self.reduce_axis_accumulate(input, monoid);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c = fold(c, input)`: [`reduce_axis`](Tile::reduce_axis) with the accumulate
    /// [`mma`](Tile::mma) carries over [`mm`](Tile::mm), folding each contracted cell into
    /// whatever `self` already holds there. That existing value is the fold's literal starting
    /// point, and the caller owns it: it must have seeded `self` with the monoid's identity
    /// ([`init_identity`](Tile::init_identity)) first, or an uninitialized accumulator folds
    /// against garbage.
    ///
    /// Also the recursion the walk re-enters per region, for the reason [`mma`](Tile::mma) gives.
    pub fn reduce_axis_accumulate<In: Numeric>(
        &mut self,
        input: &Tile<In>,
        #[comptime] monoid: Monoid,
    ) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => reduce_leaf(self, input, monoid),
            Partitioner::Level(level) => {
                let op_space = self.reduce_op_space(input);
                self.reduce_buffered(
                    input,
                    monoid,
                    op_space,
                    comptime!(level.buffering().depth()),
                );
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
    #[comptime] monoid: Monoid,
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
            reduce::memory(g, input, space, monoid);
        }
        TileKind::PlaneTile(t) => {
            reduce_plane_tile(t, input, space, monoid);
        }
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "reduce_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            reduce_plane_tile(&mut t, input, space, monoid);
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
    #[comptime] monoid: Monoid,
) {
    match tile {
        PlaneTile::Register(d) => {
            reduce::register_data(d, input, acc_space, monoid);
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
