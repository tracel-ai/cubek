use crate::kernels::tiled::BiasTile;
use cubecl::tile::{Dimension, Partitioning, SoftwareMatmul, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::plane::plane_window_microkernel;

/// Every plane of the launch, each of its units contracting its block of the output against the
/// gathered input and the rhs, read straight from global memory. The block is written as a memory
/// window, whose leaf walks every contracted dimension itself: a register accumulator cannot step a
/// gathered operand. A plane whose box starts past the output on any bounded dimension contracts
/// nothing. A bias is added to a plane's box once its units are done.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "the index also reads the extents, a runtime sequence"
)]
#[cube(launch_unchecked)]
pub fn plane_window_conv_kernel<EA: Numeric, EL: Numeric, EO: Float>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    bias: ComptimeOption<&TileTensor<'static>>,
    partitioning: Partitioning,
    #[comptime] block: SoftwareMatmul,
    #[comptime] bounded: Vec<Dimension>,
    #[comptime] bias_dimensions: Vec<Dimension>,
    extents: Sequence<usize>,
    #[define(EA, EL, EO)] _elements: [ElemType; 3],
) {
    let input = input.tile(&partitioning);
    let rhs = rhs.tile(&partitioning);
    let out = out.tile_mut(&partitioning);
    let bias = bias.map(|bias| BiasTile::new::<EO>(bias, &partitioning, bias_dimensions));

    for plane in &partitioning {
        let mut inside = true;
        #[unroll]
        for i in 0..bounded.len() {
            let dimension = comptime!(bounded[i]);
            inside = inside && plane.origin(dimension) < *extents.index(i);
        }
        if inside {
            plane_window_microkernel(plane.clone(), &input, &rhs, &out, block);
        }
        BiasTile::wait_for_plane(&bias);
        BiasTile::add_inside::<EO>(&bias, &out, &plane, inside);
    }
}
