use cubecl::ir::ElemType;
use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::rms_norm::cube::{
    CubeGroupsRmsNormMicrokernel, cube_groups_rms_norm_microkernel,
};
use cubek_microkernel::rms_norm::{RmsNormScale, RowReader};
use cubek_microkernel::row_store::CubeStage;

/// Every cube, running its groups' rows, written out as `stage` says.
#[cube(launch_unchecked)]
pub fn cube_groups_rms_norm_kernel<E: Float, H: Float, W: Float, N: Size, O: Destination>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &O::Arg<W>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeGroupsRmsNormMicrokernel,
    #[comptime] groups: usize,
    #[comptime] stage: CubeStage,
    #[define(E)] _element: ElemType,
    #[define(H)] _held_element: ElemType,
    #[define(W)] _output_element: ElemType,
) {
    let x = input.tile(&partitioning);
    let w = weight.tile(&partitioning);
    let z = gate.map(|gate| gate.tile(&partitioning));
    let o = O::tile::<W, N>(out, &partitioning);
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for cube in &partitioning {
        cube_groups_rms_norm_microkernel::<E, H, W, N>(
            &cube,
            &x.at(&cube),
            &RowReader::windowed(&z, &cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            groups,
            stage.clone(),
        );
    }
    O::finish::<W>(out);
}
