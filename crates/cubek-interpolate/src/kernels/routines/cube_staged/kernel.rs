use crate::kernels::coordinate::Rational;
use crate::kernels::filter::{SeparableFilterFamily, separable_weights};
use cubecl::tile::{Partitioning, StageStorage, Stages, TileTensor};
use cubecl::{ir::ElemType, prelude::*};
use cubek_microkernel::interpolate::cube::{
    CubeInterpolateMicrokernel, cube_interpolate_microkernel,
};

/// Every cube of the launch, resampling its box of the output one channel block at a time, each
/// block's tap window staged in shared memory once and read there by every tap. Where `padded`,
/// the stage lays the channels out in lines of that width.
#[cube(launch)]
pub fn cube_staged_interpolate_kernel<E: Float, F: SeparableFilterFamily>(
    input: &TileTensor<'_>,
    output: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] row: Rational,
    #[comptime] col: Rational,
    #[comptime] padded: Option<usize>,
    #[comptime] microkernel: CubeInterpolateMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let input = input.tile(&partitioning);
    let weights = separable_weights::<E, F>(&partitioning, row, col);
    let output = output.tile(&partitioning);
    for cube in partitioning {
        let blocks = cube.walk();
        let mut stages =
            Stages::smem_single_at(&blocks, &input, StageStorage::Strided, padded, 1usize, None);
        stages.pipelined(blocks, |slot, block| {
            let output_block = output.at(block);
            let weights_block = weights.at(block);
            slot.consume(|input_block| {
                cube_interpolate_microkernel(
                    block,
                    &output_block,
                    &weights_block,
                    input_block,
                    microkernel,
                );
            });
        });
    }
}
