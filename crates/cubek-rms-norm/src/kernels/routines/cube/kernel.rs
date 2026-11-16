use cubecl::ir::ElemType;
use cubecl::tile::launch::{Destination, ServedDestination, ServedDestinationExpand};
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::cube::{CubeRmsNormMicrokernel, cube_rms_norm_microkernel};
use cubek_microkernel::rms_norm::{GatedRow, RmsNormScale};
use cubek_microkernel::row_store::CubeStage;

/// Every cube, running its row, written out as `stage` says.
#[cube(launch_unchecked)]
pub fn cube_rms_norm_kernel<E: Float, H: Float, W: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &O::Arg<W>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeRmsNormMicrokernel,
    #[comptime] stage: CubeStage,
    #[define(E)] _element: ElemType,
    #[define(H)] _held_element: ElemType,
    #[define(W)] _output_element: ElemType,
) {
    let x = input.tile(&partitioning);
    let w = weight.tile(&partitioning);
    let z = gate.map(|gate| gate.tile(&partitioning));
    let dest = O::serve::<W, N>(out, &partitioning);
    let o = dest.tile();
    let scale = RmsNormScale::new(epsilon, microkernel.plane.dim);
    for cube in &partitioning {
        cube_rms_norm_microkernel::<E, H, W, N>(
            &cube,
            &x.at(&cube),
            &GatedRow::windowed(&z, &cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            stage.clone(),
        );
    }
    dest.finish();
}
