use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::linear_recurrence::conv::cube::{
    CubeQkNormedConvMicrokernel, QkNormedConvTiles, cube_qk_normed_conv_microkernel,
};

/// Every cube convolves one key head of one row, or of one position of a sequence, and normalizes
/// its queries and keys. `window` and `lengths` are bound over a window alone, and `columns` is the
/// ring's length there.
#[cube(launch_unchecked)]
pub fn cube_qk_normed_conv_kernel<E: Float, N: Size>(
    qkv: &TileTensor<'_>,
    filters: &TileTensor<'_>,
    window: ComptimeOption<&TileTensor<'static>>,
    lengths: ComptimeOption<&TileTensor<'static>>,
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    columns: usize,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeQkNormedConvMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let tiles = QkNormedConvTiles {
        qkv: qkv.tile(&partitioning),
        filters: filters.tile(&partitioning),
        window: window.map(|window| window.tile(&partitioning)),
        lengths: lengths.map(|lengths| lengths.tile(&partitioning)),
        query: query.tile(&partitioning),
        key: key.tile(&partitioning),
        value: value.tile(&partitioning),
        columns,
    };
    for cube in &partitioning {
        let mut tiles = tiles.at(&cube);
        cube_qk_normed_conv_microkernel::<E, N>(&cube, &mut tiles, epsilon, microkernel);
    }
}
