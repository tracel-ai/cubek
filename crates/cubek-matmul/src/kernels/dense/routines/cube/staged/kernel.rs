use crate::microkernel::dense::cube::{CubeStagedWalk, cube_staged_microkernel};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{Partitioning, StageStorage, TileArg, launch::Destination};

/// Every cube running its box, walked as `walk` says. Each operand's values are a buffer the cube's
/// units fill the stages from, or a tensor map the TMA engine fills them from; `O` is how the
/// output is reached: a buffer, or a sink.
///
/// A packed weight the walk dequantizes is read by each plane at the element its window contracts,
/// and a stage the fill decodes holds that element. An NVFP4 operand carries the words of its block
/// scales, which the stages keep beside its values and the block-scaled instruction reads with
/// them, and its per-tensor scale, one `f32` taken out for the sum and applied once, as it is
/// written.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
pub fn cube_staged_matmul_kernel<
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    E: Numeric,
    VO: Size,
    O: Destination,
>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
    #[define(EA, EL, ER, E)] _elements: [ElemType; 4],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    for cube in &partitioning {
        cube_staged_microkernel::<EA, EL>(
            cube.walk(),
            &a.at(&cube),
            &b.at(&cube),
            &c.at(&cube),
            walk,
            comptime!(storage.clone()),
        );
    }
}
