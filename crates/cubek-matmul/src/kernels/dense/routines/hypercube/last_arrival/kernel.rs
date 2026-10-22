use crate::microkernel::dense::hypercube::{
    HypercubeStagedMicrokernel, hypercube_staged_last_arrival_microkernel,
};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{Partitioning, StageStorage, TileArg, launch::Destination};

/// Every cube running its run of the line: for each tile the run touches, its part of the tile's
/// `K`, a stage at a time into one sum, handed on to the output by the tile's last arrival
/// ([`LastArrival`](cubek_tile::launch::LastArrival)). A cube whose run is empty touches no tile
/// and leaves at once.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_last_arrival_matmul_kernel<
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    E: Numeric,
    VO: Size,
    O: Destination,
>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    slots: &TileArg<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeStagedMicrokernel,
    #[comptime] storage: StageStorage,
    #[define(EA, EL, ER, E)] _elements: [ElemType; 4],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    // The walk under the cubes steps a tile's `K` a stage at a time: what a run is counted in.
    let stages = comptime!(partitioning.levels()[1].clone());
    let portion = partitioning.walk().portion(comptime!(stages.clone()));
    for i in 0..portion.touched() {
        let tile = portion.region(i);
        let (from, steps) = portion.steps(i);
        let merge = slots.last_arrival::<VO>(&partitioning, &portion, i, arrivals);
        hypercube_staged_last_arrival_microkernel::<EA, EL>(
            tile.over(&stages).range(from, steps),
            &a.at(&tile),
            &b.at(&tile),
            merge,
            &c,
            comptime!(microkernel.cube.walk()),
            comptime!(storage.clone()),
        );
    }
}
