use cubecl::ir::ElemType;
use cubecl::tile::launch::{Destination, ServedDestination, ServedDestinationExpand};
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::plane::{PlaneRmsNormMicrokernel, plane_rms_norm_microkernel};
use cubek_microkernel::rms_norm::{Gate, GatedRow, RmsNormScale};
use cubek_microkernel::row_store::OwnerLines;

/// Every plane of the launch, running its row.
#[cube(launch_unchecked)]
pub fn plane_rms_norm_kernel<E: Float, Held: Float, Out: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &O::Arg<Out>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
    #[define(Held)] _held_element: ElemType,
    #[define(Out)] _output_element: ElemType,
) {
    let gated = GatedRow::new(
        &input.tile(&partitioning),
        Gate::bound(gate.map(|gate| gate.tile(&partitioning)), microkernel.gate),
    );
    let w = weight.tile(&partitioning);
    let dest = O::serve::<Out, N>(out, &partitioning);
    let o = dest.tile();
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for plane in &partitioning {
        let mut row_out = OwnerLines::window(o.at(&plane));
        plane_rms_norm_microkernel::<E, Held, Out, N>(
            &gated.at(&plane),
            &w.at(&plane),
            &mut row_out,
            &scale,
            microkernel,
        );
    }
    dest.finish();
}
