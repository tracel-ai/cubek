use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::linear_recurrence::conv::cube::{
    ChannelBox, PastTiles, cube_ring_write_back_microkernel,
};

/// Every cube copies a box of channel lines of one of a row's last `taps - 1` valid positions of a
/// prompt into its ring. It runs after the convolution has read every ring.
#[cube(launch_unchecked)]
pub fn ring_write_back_kernel<E: Float, N: Size>(
    input: &TileTensor<'_>,
    ring: &TileTensor<'_>,
    lengths: &TileTensor<'_>,
    valid: &TileTensor<'_>,
    columns: usize,
    partitioning: Partitioning,
    #[comptime] channel_box: ChannelBox,
    #[comptime] read_columns: usize,
    #[define(E)] _element: ElemType,
) {
    let tiles = PastTiles {
        input: input.tile(&partitioning),
        ring: ring.tile(&partitioning),
        lengths: lengths.tile(&partitioning),
        columns,
    };
    let valid = valid.tile(&partitioning);
    for cube in &partitioning {
        let tiles = tiles.at(&cube);
        let valid = valid.at(&cube);
        cube_ring_write_back_microkernel::<E, N>(&cube, &tiles, &valid, channel_box, read_columns);
    }
}
