use crate::matmul::MatmulOperands;
use crate::matmul::plane::{PlaneShuffleFetch, plane_shuffle_walk};
use cubecl::tile::launch::LastArrival;
use cubecl::tile::{Instruction, Semiring, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a box any number of runs share, the cube one plane of the shuffle: its
/// units walk the cube's run of `K` into one register sum opened over its slot of the box, each
/// step's loads issued as `fetch` says ([`plane_shuffle_walk`]), and the sum, under the scales the
/// operands give up to it, is handed on straight into `out` where this run holds the box whole,
/// for the box's last arrival to add up otherwise ([`LastArrival`]).
#[cube]
pub fn hypercube_shuffle_last_arrival_microkernel<'a, EA: Numeric>(
    run: Walk,
    a: &Tile,
    b: &Tile,
    merge: LastArrival<'a>,
    out: &Tile,
    #[comptime] instruction: Instruction,
    #[comptime] fetch: PlaneShuffleFetch,
) {
    let mut out = out.clone();
    let slot = merge.tile();
    let (a, a_factor) = a.factor(slot.space(), instruction);
    let (b, b_factor) = b.factor(slot.space(), instruction);
    let mut sum = slot.accumulator::<EA>(&a, &b, instruction, Semiring::SUM_PROD);
    sum.zero();
    plane_shuffle_walk(run, &sum, &a, &b, MatmulOperands::AS_BOUND, fetch);
    merge.hand_on(&sum.scaled(&a_factor).scaled(&b_factor), &mut out);
}
