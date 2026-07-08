//! The cmma boundary hoist: a memory accumulator bound for the [`Leaf::Cmma`] instruction
//! becomes a plane-resident fragment *before* the level walk — initialized from this
//! instance's window, accumulated across the whole contraction, drained back after. A
//! fragment cannot be revisited per region the way a memory window can (no runtime
//! indexing into fragments), so residency is handled once here, not per schedule.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// Hoist `out` (a memory tile) into a resident accumulator fragment, initialized from
/// this instance's final window so the contraction accumulates onto the delivered values.
/// The fragment carries `out`'s space, so the schedules walk it exactly like the tile it
/// replaces; [`Tile::at`] passes it through unchanged.
#[cube]
pub(crate) fn cmma_acc<Acc: Numeric, Lhs: Numeric>(
    out: &mut Tile<Acc>,
    lhs: &Tile<Lhs>,
) -> Tile<Acc> {
    let space = comptime!(out.space.clone());
    comptime!(assert!(
        single_tile_per_instance(&space),
        "cmma accumulator: every remaining level must hand each instance exactly one tile"
    ));
    let fin = comptime!(space.final_space());
    let m = comptime!(fin.extent_at(fin.rank() - 2));
    let n = comptime!(fin.extent_at(fin.rank() - 1));
    let k = comptime!(contracted_extent(&lhs.space, &space));

    let mut acc = Tile::<Acc>::cmma_fragment(
        MatrixIdent::Accumulator,
        m,
        n,
        k,
        MatrixLayout::RowMajor,
        comptime!(space.clone()),
    );
    let window = instance_window(out);
    acc.copy_from(&window);
    acc
}

/// Drain the resident fragment back into this instance's final window of `out`.
#[cube]
pub(crate) fn cmma_drain<Acc: Numeric>(out: &mut Tile<Acc>, acc: &Tile<Acc>) {
    let mut window = instance_window(out);
    window.copy_from(acc);
}

/// Descend to this instance's final window: every remaining level hands this instance a
/// single region (asserted at the hoist), and `region(0)` folds the instance's hardware
/// position in, so the chain of `at`s lands on the one window this instance owns.
#[cube]
fn instance_window<T: Numeric>(tile: &mut Tile<T>) -> Tile<T> {
    let walk = Walk::over(tile.runtime_space());
    let mut sub = tile.at(&walk.region(0));
    match comptime!(sub.space.partitioner()) {
        Partitioner::Final(_) => sub,
        Partitioner::Level(_) => instance_window(&mut sub),
    }
}

/// The contraction depth `k`: the final-space extent of the operand axis the output drops.
fn contracted_extent(operand: &Space, out: &Space) -> usize {
    let contracted = operand.contracting(out);
    assert!(
        contracted.len() == 1,
        "cmma accumulator: the leaf contracts exactly one axis"
    );
    operand.final_space().extent(contracted[0])
}

/// Whether every remaining level hands each instance exactly one tile on every axis of
/// `space` — the invariant that lets one resident fragment serve the whole walk.
fn single_tile_per_instance(space: &Space) -> bool {
    let mut level = space.clone();
    while !level.is_final() {
        for axis in level.axes() {
            let edge = level.partitioner().edge(axis);
            let tiles = match level.partitioner().distribution(axis) {
                Distribution::Sequential => match level.extent_raw(axis) {
                    Extent::Static(e) => e.div_ceil(edge),
                    // A dynamic sequential axis walks a runtime tile count.
                    Extent::Dynamic => return false,
                },
                Distribution::Spatial { coverage, .. } => match coverage {
                    Coverage::TilesEach(t) => t,
                    Coverage::Instances(n) => match level.extent_raw(axis) {
                        Extent::Static(e) => e.div_ceil(edge).div_ceil(n),
                        Extent::Dynamic => return false,
                    },
                },
            };
            if tiles != 1 {
                return false;
            }
        }
        level = level.divide();
    }
    true
}
