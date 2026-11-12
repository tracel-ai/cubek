use crate::kernels::coordinate::Rational;
use crate::kernels::filter::{SeparableFilterFamily, separable_weights};
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{ir::ElemType, prelude::*};
use cubek_microkernel::interpolate::cube::{
    CubeInterpolateMicrokernel, cube_interpolate_microkernel,
};

/// Every cube of the launch, resampling its box of the output one channel block at a time, each
/// block's taps read from the input where it lies.
#[cube(launch)]
pub fn cube_interpolate_kernel<E: Float, F: SeparableFilterFamily>(
    input: &TileTensor<'_>,
    output: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] row: Rational,
    #[comptime] col: Rational,
    #[comptime] microkernel: CubeInterpolateMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let input = input.tile(&partitioning);
    let weights = separable_weights::<E, F>(&partitioning, row, col);
    let output = output.tile(&partitioning);
    for cube in partitioning {
        for block in cube.walk() {
            cube_interpolate_microkernel(
                &block,
                &output.at(&block),
                &weights.at(&block),
                &input.at(&block),
                microkernel,
            );
        }
    }
}
