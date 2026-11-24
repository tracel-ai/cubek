use super::base::CubeLineReduceMicrokernel;
use crate::reduce::cube::{CubeStridedReduceMicrokernel, cube_strided_reduce_microkernel};
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, one line wide: `cube`'s region of `input`, reduced into `out` and `indices` at
/// the same region. The cube reduces its window as [`CubeStridedReduceMicrokernel`] does and its
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
    let mut rows_out = out.at(cube);
    let rows_indices = indices.clone().map(|tile| tile.at(cube));
    if UNIT_POS == 0 {
        unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
            total,
            &mut rows_out,
            &rows_indices,
            count * row_lanes,
            comptime!(UnitCarrierWriteMicrokernel::new(
                microkernel.item(),
                row_lanes
            )),
        );
    }
}
