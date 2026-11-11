use cubecl::ir::ElemType;
use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::RmsNormScale;
use cubek_microkernel::rms_norm::cube::{CubeRmsNormMicrokernel, cube_rms_norm_microkernel};
use cubek_microkernel::row_store::CubeStage;

/// Every cube, running its row, written out as `stage` says.
#[cube(launch_unchecked)]
pub fn cube_rms_norm_kernel<E: Float, W: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &O::Arg<W>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeRmsNormMicrokernel,
    #[comptime] stage: CubeStage,
    #[define(E)] _element: ElemType,
    #[define(W)] _output_element: ElemType,
) {
    let x = input.tile(&partitioning);
    let w = weight.tile(&partitioning);
    let o = O::tile::<W, N>(out, &partitioning);
    let scale = RmsNormScale::new(epsilon, microkernel.plane.dim);
    for cube in &partitioning {
        cube_rms_norm_microkernel::<E, W, N>(
            &cube,
            &x.at(&cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            stage.clone(),
        );
    }
    O::finish::<W>(out);
}
