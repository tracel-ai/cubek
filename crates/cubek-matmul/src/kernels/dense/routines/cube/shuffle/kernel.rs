use cubecl::tile::launch::Destination;
use cubecl::tile::{Axis, Instruction, Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::dense::cube::{CubeMeet, cube_shuffle_microkernel};
use cubek_microkernel::matmul::{MatmulOperands, MatmulRead};

/// Every cube, running its run of the output, its units reading each operand as `reads` says,
/// under the scales it carries, none where it is plain. `O` is how the output is reached: a buffer, or a
/// sink.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_shuffle_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] instruction: Instruction,
    #[comptime] meet: CubeMeet,
    #[comptime] split: Axis,
    #[comptime] planes: usize,
    #[comptime] reads: MatmulOperands<MatmulRead>,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    for cube in &partitioning {
        cube_shuffle_microkernel::<EA>(
            &cube,
            &c.at(&cube),
            &a.at(&cube),
            &b.at(&cube),
            instruction,
            meet,
            split,
            planes,
            reads,
        );
    }
}
