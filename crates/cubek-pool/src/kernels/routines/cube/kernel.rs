use crate::kernels::AverageCount;
use crate::kernels::space::WindowAxis;
use crate::kernels::weights::average_weights;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{ir::ElemType, prelude::*};
use cubek_microkernel::pool::PoolReduction;
use cubek_microkernel::pool::cube::{
    CubePoolMicrokernel, cube_average_pool_microkernel, cube_max_pool_microkernel,
};

/// Every cube of the launch, pooling its box of the output one channel block at a time, each
/// block's windows read from the input where it lies.
#[cube(launch)]
pub fn cube_pool_kernel<E: Float>(
    input: &TileTensor<'_>,
    output: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] windows: [WindowAxis; 2],
    #[comptime] input_size: [usize; 2],
    #[comptime] count: Option<AverageCount>,
    #[comptime] microkernel: CubePoolMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let input = input.tile(&partitioning);
    let output = output.tile(&partitioning);
    match microkernel.reduction {
        PoolReduction::Max => {
            for cube in partitioning {
                for block in cube.walk() {
                    cube_max_pool_microkernel(&block, &output.at(&block), &input.at(&block));
                }
            }
        }
        PoolReduction::Average => {
            let weights = average_weights::<E>(
                &partitioning,
                windows,
                input_size,
                comptime!(count.expect("an average names what it divides by")),
            );
            for cube in partitioning {
                for block in cube.walk() {
                    cube_average_pool_microkernel(
                        &block,
                        &output.at(&block),
                        &weights.at(&block),
                        &input.at(&block),
                        microkernel,
                    );
                }
            }
        }
    }
}
