use cubecl::tile::{Partitioning, Scratch, StageStorage, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::cube::{CubeStagedWalk, cube_staged_microkernel};

/// Every cube running the cube microkernel on its box of the output, staging a gathered input and
/// an rhs held as `storage`.
#[cube(launch_unchecked)]
pub fn cube_staged_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let input = input.tile(&partitioning);
    let rhs = rhs.tile(&partitioning);
    let out = out.tile(&partitioning);

    for cube in &partitioning {
        // Lets the window reuse the stages' bytes; safe while the trip count is uniform over the cube.
        if walk.scratch != Scratch::None {
            sync_cube();
        }
        cube_staged_microkernel::<EA, EL>(
            cube.walk(),
            &input.at(&cube),
            &rhs.at(&cube),
            &out.at(&cube),
            walk,
            storage.clone(),
        );
    }
}
