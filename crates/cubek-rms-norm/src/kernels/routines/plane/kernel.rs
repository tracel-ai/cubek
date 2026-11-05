use cubecl::ir::ElemType;
use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::RmsNormScale;
use cubek_microkernel::rms_norm::plane::{PlaneRmsNormMicrokernel, plane_rms_norm_microkernel};
use cubek_microkernel::row_store::OwnerLines;

/// Every plane of the launch, running its row.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn plane_rms_norm_kernel<E: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &O::Arg<E>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let x = input.tile(&partitioning);
    let w = weight.tile(&partitioning);
    let o = O::tile::<E, N>(out, &partitioning);
    let scale = RmsNormScale::new(epsilon, comptime!(microkernel.dim));
    for plane in &partitioning {
        let mut row_out = OwnerLines::window(o.at(&plane));
        plane_rms_norm_microkernel::<E, N>(
            &x.at(&plane),
            &w.at(&plane),
            &mut row_out,
            &scale,
            microkernel,
        );
    }
}
