use super::base::CubeLineReduceMicrokernel;
use crate::reduce::cube::{CubeStridedReduceMicrokernel, cube_strided_reduce_microkernel};
use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// The cube reduces its window as
/// [`CubeStridedReduceMicrokernel`](crate::reduce::cube::CubeStridedReduceMicrokernel) does, and its
/// first unit writes the total.
#[cube]
pub(crate) fn cube_line_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeLineReduceMicrokernel,
) {
    let rows = input.at(cube);
    let count = rows.flat::<In, N>().shape();
    let total = cube_strided_reduce_microkernel::<In, Acc, N, C>(
        &rows,
        0,
        count,
        comptime!(CubeStridedReduceMicrokernel::new(
            microkernel.item(),
            microkernel.lanes(),
            microkernel.planes()
        )),
    );

    let load = rows.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    unit_region_write_microkernel::<Out, Idx, Acc, N, C>(
        cube,
        total,
        out,
        indices,
        count * row_lanes,
        UNIT_POS == 0,
        comptime!(UnitRegionWriteMicrokernel::new(
            microkernel.item(),
            row_lanes
        )),
    );
}
