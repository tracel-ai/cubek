use crate::microkernel::dense::hypercube::{
    HypercubeStagedMicrokernel, hypercube_staged_relay_microkernel,
};
use cubecl::tile::{Partitioning, StageStorage, TileArg, launch::Destination};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every cube running its part of its box: its run of `K`, a stage at a time into one sum, then
/// its turn at the box's carry ([`Relay`](cubecl::tile::launch::Relay)), the last turn handing the
/// sum on to the output. A cube whose run holds no stage takes no turn and leaves at once. A box's
/// turns lie on the grid's third dimension, a whole grid layer apart: a waiting cube rests on the
/// earlier turns having started, which the device's dispatch of the grid in order gives.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_relay_matmul_kernel<
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
    carry: &TileArg<'_>,
    turns: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeStagedMicrokernel,
    #[comptime] storage: StageStorage,
    #[define(EA, EL, ER, E)] _elements: [ElemType; 4],
) {
    let relay = carry.relay::<VO>(&partitioning, turns);
    // Uniform across the cube, so it leaves before any barrier.
    if !relay.has_turn() {
        terminate!();
    }
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    hypercube_staged_relay_microkernel::<EA, EL>(
        partitioning.walk(),
        &a,
        &b,
        relay,
        &c,
        comptime!(microkernel.cube.walk()),
        storage,
    );
}
