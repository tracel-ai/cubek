use crate::matmul::cube::{CubeStagedWalk, cube_staged_sum_microkernel};
use cubecl::tile::launch::{LastArrival, ServedDestination};
use cubecl::tile::{StageStorage, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a box any number of runs share: the cube walks its run of `K` into one sum
/// ([`cube_staged_sum_microkernel`](fn@cube_staged_sum_microkernel)), opened over its slot of the
/// box, and hands it on to `dest` where this run holds the box whole, for the box's last arrival
/// to add up and hand on otherwise ([`LastArrival`]).
#[cube]
pub fn hypercube_staged_last_arrival_microkernel<'a, EA: Numeric, D: ServedDestination>(
    run: Walk,
    a: &Tile,
    b: &Tile,
    merge: LastArrival<'a>,
    dest: &D,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    // As a staged cube's: the scales the sum took out of its inputs are applied as each run parks
    // its part.
    let opened = CubeStagedWalk::opened::<EA>(&merge.tile(), a, b, walk);
    cube_staged_sum_microkernel(run, &opened.sum, &opened.a, &opened.b, walk, storage);

    let sum = opened.scaled(CubeStagedWalk::drain_sum(opened.sum.clone(), walk));
    merge.hand_on::<D>(&sum, dest);
}
