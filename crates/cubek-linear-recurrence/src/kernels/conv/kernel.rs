use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::linear_recurrence::conv::cube::{
    CausalConvPastTiles, CausalConvTiles, ChannelRun, CubeCausalConvMicrokernel,
    cube_causal_conv_microkernel, cube_ring_write_back_microkernel,
};

/// Every cube convolves a run of channel lines of one position of a row into `output`. `bias` is
/// bound where the conv has one, and `columns` is the ring's length.
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
        past: CausalConvPastTiles {
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

/// Every cube copies a run of channel lines of one of a row's last `taps - 1` valid positions of a
/// prompt into its ring. It runs after the convolution has read every ring.
#[cube(launch_unchecked)]
pub fn ring_write_back_kernel<E: Float, N: Size>(
    input: &TileTensor<'_>,
    ring: &TileTensor<'_>,
    lengths: &TileTensor<'_>,
    valid: &TileTensor<'_>,
    columns: usize,
    positions: usize,
    partitioning: Partitioning,
    #[comptime] channel_run: ChannelRun,
    #[comptime] read_columns: usize,
    #[define(E)] _element: ElemType,
) {
    let tiles = CausalConvPastTiles {
        input: input.tile(&partitioning),
        ring: ring.tile(&partitioning),
        lengths: lengths.tile(&partitioning),
        columns,
    };
    let valid = valid.tile(&partitioning);
    for cube in &partitioning {
        let tiles = tiles.at(&cube);
        let valid = valid.at(&cube);
        cube_ring_write_back_microkernel::<E, N>(
            &cube,
            &tiles,
            &valid,
            positions,
            channel_run,
            read_columns,
        );
    }
}
