use crate::microkernel::dense::cube::{CubeStagedWalk, cube_staged_sum_microkernel};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{
    StageStorage, Tile, Walk,
    launch::{LastArrival, Relay},
};

/// One cube's part of a box it shares with the cubes before and after it along `K`: its run of
/// `K` arrives as boxes one stage deep, the cube's regions of a partitioning that deals `K` to its
/// cubes in runs, each staged on its own and added into one sum
/// ([`cube_staged_sum_microkernel`](fn@cube_staged_sum_microkernel)). The cube then takes its turn at `relay`'s carry, the last
/// turn handing the sum on to `out` ([`Relay`]).
#[cube]
pub fn hypercube_staged_relay_microkernel<'a, EA: Numeric, EL: Numeric>(
    run: Walk,
    a: &Tile,
    b: &Tile,
    relay: Relay<'a>,
    out: &Tile,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    let mut out = out.clone();
    let mut carry = relay.tile();
    let (a, a_factor) = a.factor(carry.space(), comptime!(walk.instruction));
    let (b, b_factor) = b.factor(carry.space(), comptime!(walk.instruction));
    let first = run.region(0usize);
    let sum = CubeStagedWalk::open_sum::<EA>(&carry, &a.at(&first), &b.at(&first), walk);
    for stage_box in run {
        cube_staged_sum_microkernel::<EL>(
            stage_box.walk(),
            &sum,
            &carry,
            &a.at(&stage_box),
            &b.at(&stage_box),
            walk,
            comptime!(storage.clone()),
        );
    }
    relay.take();
    let sum = CubeStagedWalk::drain_sum(sum, walk);
    carry.dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
    relay.pass(&mut out);
}

/// One cube's part of a box any number of runs share: the cube walks its run of `K` into one sum
/// ([`cube_staged_sum_microkernel`](fn@cube_staged_sum_microkernel)), opened over its slot of the box, and hands it on straight
/// into `out` where this run holds the box whole, for the box's last arrival to add up otherwise
/// ([`LastArrival`]).
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
