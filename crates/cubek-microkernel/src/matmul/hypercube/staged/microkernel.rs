use crate::matmul::cube::{CubeStagedWalk, cube_staged_sum_microkernel};
use cubecl::tile::launch::LastArrival;
use cubecl::tile::{StageStorage, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a box any number of runs share: the cube walks its run of `K` into one sum
/// ([`cube_staged_sum_microkernel`](fn@cube_staged_sum_microkernel)), opened over its slot of the
/// box, and hands it on straight into `out` where this run holds the box whole, for the box's last
/// arrival to add up otherwise ([`LastArrival`]).
#[cube]
pub fn hypercube_staged_last_arrival_microkernel<'a, EA: Numeric, EL: Numeric>(
    run: Walk,
    a: &Tile,
    b: &Tile,
    merge: LastArrival<'a>,
    out: &Tile,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    let mut out = out.clone();
    let slot = merge.tile();

    let sum = CubeStagedWalk::open_sum::<EA>(&slot, a, b, walk);
    cube_staged_sum_microkernel::<EL>(run, &sum, &slot, a, b, walk, storage);

    let sum = CubeStagedWalk::drain_sum(sum, walk);
    merge.hand_on(&sum, &mut out);
}
