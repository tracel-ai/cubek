use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::linear_recurrence::conv::cube::{
    CausalConvTiles, CubeCausalConvMicrokernel, PastTiles, cube_causal_conv_microkernel,
};

/// Every cube convolves a box of channel lines of one row, or of one position of a prompt, into
/// `output`. `bias` is bound where the conv has one, and `columns` is the ring's length.
#[cube(launch_unchecked)]
pub fn cube_causal_conv_kernel<E: Float, O: Float, N: Size>(
    input: &TileTensor<'_>,
    filters: &TileTensor<'_>,
    bias: ComptimeOption<&TileTensor<'static>>,
    ring: &TileTensor<'_>,
    lengths: &TileTensor<'_>,
    output: &TileTensor<'_>,
    columns: usize,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeCausalConvMicrokernel,
    #[define(E)] _element: ElemType,
    #[define(O)] _output_element: ElemType,
) {
    let tiles = CausalConvTiles {
        past: PastTiles {
            input: input.tile(&partitioning),
            ring: ring.tile(&partitioning),
            lengths: lengths.tile(&partitioning),
            columns,
        },
        filters: filters.tile(&partitioning),
        bias: bias.map(|bias| bias.tile(&partitioning)),
        output: output.tile(&partitioning),
    };
    for cube in &partitioning {
        let mut tiles = tiles.at(&cube);
        cube_causal_conv_microkernel::<E, O, N>(&cube, &mut tiles, microkernel);
    }
}
