use cubecl::tile::{Partitioning, Scratch, Semiring, StageStorage, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::plane::{
    PlaneFragmentMicrokernel, plane_fragment_staged_microkernel,
};
use cubek_std::microkernel::StageRing;

/// Every plane running the plane microkernel on its box of the output, staging a gathered input
/// and an rhs held as `storage`.
#[cube(launch_unchecked)]
pub fn plane_fragment_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &TileArg<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] storage: StageStorage,
    #[comptime] ring: StageRing,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let input = input.tile(&partitioning);
    let rhs = rhs.tile(&partitioning);
    let out = out.tile(&partitioning);

    for plane in &partitioning {
        let (a, b, c) = (input.at(&plane), rhs.at(&plane), out.at(&plane));
        let mut sum = c
            .accumulator::<EA>(&a, &b, microkernel.tile_instruction(), Semiring::SUM_PROD)
            .with_scratch(Scratch::None);
        sum.zero();
        plane_fragment_staged_microkernel::<EL>(
            &plane,
            &sum,
            &c,
            &a,
            &b,
            microkernel,
            storage.clone(),
            ring,
        );
        sum.drained_into(&c);
    }
}
