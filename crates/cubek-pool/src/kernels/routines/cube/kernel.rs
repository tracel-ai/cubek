use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{ir::ElemType, prelude::*};
use cubek_microkernel::pool::cube::cube_max_pool_microkernel;

/// Every cube of the launch, max pooling its box of the output one channel block at a time, each
/// block's windows read from the input where it lies.
#[cube(launch)]
pub fn cube_max_pool_kernel<E: Numeric>(
    input: &TileTensor<'_>,
    output: &TileTensor<'_>,
    partitioning: Partitioning,
    #[define(E)] _element: ElemType,
) {
    let input = input.tile(&partitioning);
    let output = output.tile(&partitioning);
    for cube in partitioning {
        for block in cube.walk() {
            cube_max_pool_microkernel(&block, &output.at(&block), &input.at(&block));
        }
    }
}
