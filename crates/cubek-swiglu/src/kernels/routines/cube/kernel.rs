use cubecl::ir::ElemType;
use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::row_store::CubeStage;
use cubek_microkernel::swiglu::cube::{CubeSwigluMicrokernel, cube_swiglu_microkernel};

/// Every cube, running its groups' runs, written out as `stage` says.
#[cube(launch_unchecked)]
pub fn cube_swiglu_kernel<E: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeSwigluMicrokernel,
    #[comptime] stage: CubeStage,
    #[define(E)] _element: ElemType,
) {
    let x = input.tile(&partitioning);
    let o = O::tile::<E, N>(out, &partitioning);
    for cube in &partitioning {
        cube_swiglu_microkernel::<E, N>(
            &cube,
            &x.at(&cube),
            &o.at(&cube),
            microkernel,
            comptime!(stage.clone()),
        );
    }
}
