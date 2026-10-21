use super::base::PlaneFragmentMatmulRoutineSharedMemory;

use crate::microkernel::dense::plane::{
    PlaneFragmentMicrokernel, plane_fragment_staged_microkernel,
};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{Partitioning, Semiring, TileArg, launch::Destination};

/// Every plane of the launch owning its tile: it walks the tile through stages of its own
/// ([`plane_fragment_staged_microkernel`]) and drains its fragments into the output, through the
/// scratch where it holds a window.
#[cube(launch_unchecked)]
pub fn plane_fragment_matmul_kernel<
    EA: Numeric,
    EL: Numeric,
    E: Numeric,
    VO: Size,
    O: Destination,
>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] shared_memory: PlaneFragmentMatmulRoutineSharedMemory,
    #[define(EA, EL, E)] _elements: [ElemType; 3],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    for plane in &partitioning {
        let (c, a, b) = (c.at(&plane), a.at(&plane), b.at(&plane));
        let mut sum = c
            .accumulator::<EA>(
                &a,
                &b,
                comptime!(microkernel.tile_instruction()),
                Semiring::SUM_PROD,
            )
            .with_scratch(comptime!(shared_memory.scratch));
        sum.zero();
        plane_fragment_staged_microkernel::<EL>(
            &plane,
            &sum,
            &c,
            &a,
            &b,
            microkernel,
            comptime!(shared_memory.stage.clone()),
            comptime!(shared_memory.ring),
        );
        sum.drained_into(&c);
    }
}
