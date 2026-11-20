use crate::matmul::cube::{CubeStagedWalk, cube_staged_sum_microkernel};
use cubecl::tile::launch::{LastArrival, ServedDestination};
use cubecl::tile::{StageStorage, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a box any number of runs share: the cube walks its run of `K` into one sum
/// ([`cube_staged_sum_microkernel`](fn@cube_staged_sum_microkernel)), opened over its slot of the
/// box on the inputs as its stages hold them ([`CubeStagedWalk::as_staged`]), and hands it on to
/// `dest` where this run holds the box whole, for the box's last arrival to add up and hand on
/// otherwise ([`LastArrival`]).
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
    let slot = merge.tile();

    let (a, b) = (
        CubeStagedWalk::as_staged(a, walk),
        CubeStagedWalk::as_staged(b, walk),
    );
    let sum = CubeStagedWalk::open_sum::<EA>(&slot, &a, &b, walk);
    cube_staged_sum_microkernel(run, &sum, &a, &b, walk, storage);

    let sum = CubeStagedWalk::drain_sum(sum, walk);
    merge.hand_on::<D>(&sum, dest);
}
