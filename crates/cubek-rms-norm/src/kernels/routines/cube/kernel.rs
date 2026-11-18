use cubecl::ir::ElemType;
use cubecl::tile::launch::{Destination, ServedDestination, ServedDestinationExpand};
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::cube::{CubeRmsNormMicrokernel, cube_rms_norm_microkernel};
use cubek_microkernel::rms_norm::{Gate, GatedRow, RmsNormScale};
use cubek_microkernel::row_store::CubeStage;

/// Every cube, running its row, written out as `stage` says.
#[cube(launch_unchecked)]
pub fn cube_rms_norm_kernel<E: Float, Held: Float, Out: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &O::Arg<Out>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeRmsNormMicrokernel,
    #[comptime] stage: CubeStage,
    #[define(E)] _element: ElemType,
    #[define(Held)] _held_element: ElemType,
    #[define(Out)] _output_element: ElemType,
) {
    let gated = GatedRow::new(
        &input.tile(&partitioning),
        Gate::bound(
            gate.map(|gate| gate.tile(&partitioning)),
            microkernel.plane.gate,
        ),
    );
    let w = weight.tile(&partitioning);
    let dest = O::serve::<Out, N>(out, &partitioning);
    let o = dest.tile();
    let scale = RmsNormScale::new(epsilon, microkernel.plane.dim);
    for cube in &partitioning {
        cube_rms_norm_microkernel::<E, Held, Out, N>(
            &cube,
            &gated.at(&cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            stage.clone(),
        );
    }
    dest.finish();
}
