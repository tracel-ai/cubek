use cubecl::ir::ElemType;
use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::plane::{PlaneRmsNormMicrokernel, plane_rms_norm_microkernel};
use cubek_microkernel::rms_norm::{RmsNormScale, RowReader};
use cubek_microkernel::row_store::OwnerLines;

/// Every plane of the launch, running its row.
#[cube(launch_unchecked)]
pub fn plane_rms_norm_kernel<E: Float, H: Float, W: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &O::Arg<W>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
    #[define(H)] _held_element: ElemType,
    #[define(W)] _output_element: ElemType,
) {
    let x = input.tile(&partitioning);
    let w = weight.tile(&partitioning);
    let z = gate.map(|gate| gate.tile(&partitioning));
    let o = O::tile::<W, N>(out, &partitioning);
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for plane in &partitioning {
        let mut row_out = OwnerLines::window(o.at(&plane));
        plane_rms_norm_microkernel::<E, H, W, N>(
            &x.at(&plane),
            &RowReader::windowed(&z, &plane),
            &w.at(&plane),
            &mut row_out,
            &scale,
            microkernel,
        );
    }
    O::finish::<W>(out);
}
