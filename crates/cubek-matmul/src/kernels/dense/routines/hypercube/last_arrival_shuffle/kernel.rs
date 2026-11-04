use cubecl::tile::launch::Destination;
use cubecl::tile::{Instruction, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::hypercube::hypercube_shuffle_last_arrival_microkernel;
use cubek_microkernel::matmul::plane::PlaneShuffleFetch;

/// Every cube, one plane of the shuffle, running its run of the line: for each box the run
/// touches, its part of the box's `K`, a step at a time into one register sum, handed on to the
/// output by the box's last arrival ([`LastArrival`](cubecl::tile::launch::LastArrival)). A cube
/// whose run is empty touches no box and leaves at once.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_last_arrival_shuffle_matmul_kernel<
    EA: Numeric,
    E: Numeric,
    VO: Size,
    O: Destination,
>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &O::Arg<E>,
    slots: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] instruction: Instruction,
    #[comptime] fetch: PlaneShuffleFetch,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);

    // The walk under the cubes steps a box's `K` a step of the plane at a time: what a run is
    // counted in.
    let steps = comptime!(partitioning.levels()[1].clone());
    let portion = partitioning.walk().portion(steps.clone());

    for i in 0..portion.touched() {
        let tile = portion.region(i);
        let (from, count) = portion.steps(i);
        let merge = slots.last_arrival::<VO>(&partitioning, &portion, i, arrivals);
        hypercube_shuffle_last_arrival_microkernel::<EA>(
            tile.over(&steps).range(from, count),
            &a.at(&tile),
            &b.at(&tile),
            merge,
            &c,
            instruction,
            fetch,
        );
    }
}
