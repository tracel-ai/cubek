use crate::kernels::tiled::BiasTile;
use cubecl::tile::{Dimension, Partitioning, StageStorage, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::cube::{CubeStagedDrain, CubeStagedWalk, cube_staged_microkernel};

/// Every cube running the cube microkernel on its box of the output, staging a gathered input and
/// an rhs held as `storage`, then adding the channel bias to the tiles its planes wrote.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
pub fn cube_staged_conv_kernel<EA: Numeric, EL: Numeric, EO: Float>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    bias: ComptimeOption<&TileTensor<'static>>,
    partitioning: Partitioning,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
    #[comptime] bias_dimensions: Vec<Dimension>,
    #[define(EA, EL, EO)] _elements: [ElemType; 3],
) {
    let input = input.tile(&partitioning);
    let rhs = rhs.tile(&partitioning);
    let out = out.tile_mut(&partitioning);
    let bias = bias.map(|bias| BiasTile::new::<EO>(bias, &partitioning, bias_dimensions));

    // No barrier between boxes: each box's first stage refill synchronizes the cube.
    for cube in &partitioning {
        cube_staged_microkernel::<EA, Tile>(
            cube.walk(),
            &input.at(&cube),
            &rhs.at(&cube),
            &out,
            &cube,
            walk,
            storage.clone(),
        );
        // Merged partials reach the output through the whole cube, so a plane reads back cells it
        // may not have written.
        if comptime!(walk.drain == CubeStagedDrain::Merged) {
            BiasTile::wait_for_cube(&bias);
        }
        for plane in cube.planes() {
            BiasTile::add_to::<EO>(&bias, &out, &plane);
        }
    }
}
