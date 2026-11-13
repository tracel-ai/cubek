use cubecl::tile::launch::Destination;
use cubecl::tile::{Axis, Partitioning, SoftwareMatmul, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::cube::{CubeMeet, cube_shuffle_microkernel};
use cubek_microkernel::matmul::{MatmulOperands, MatmulRead};

/// Every cube, running its run of the output, its units reading each operand as `reads` says, under
/// the scales it carries, none where it is plain. `O` is how the output is reached: a buffer, or a
/// sink. A `persistent` launch's cubes each walk a run of the boxes rather than take one.
#[cube(launch_unchecked)]
pub fn cube_shuffle_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] block: Option<SoftwareMatmul>,
    #[comptime] meet: CubeMeet,
    #[comptime] split: Axis,
    #[comptime] planes: usize,
    #[comptime] reads: MatmulOperands<MatmulRead>,
    #[comptime] persistent: bool,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);

    // A persistent cube's run of the boxes, whole boxes each, the cube level sharing them as one
    // index; every other cube's one box.
    let cubes = if persistent {
        partitioning.walk().portion_of(1usize).regions()
    } else {
        partitioning.walk()
    };
    for cube in cubes {
        cube_shuffle_microkernel::<EA>(
            &cube,
            &c.at(&cube),
            &a.at(&cube),
            &b.at(&cube),
            block,
            meet,
            split,
            planes,
            reads,
        );
    }
    O::finish::<E>(out);
}
