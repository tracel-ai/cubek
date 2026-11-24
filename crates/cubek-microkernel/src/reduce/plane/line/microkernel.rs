use super::base::PlaneLineReduceMicrokernel;
use crate::reduce::read::{UnitStridedReduceMicrokernel, unit_strided_reduce_microkernel};
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's rows, one line wide: `plane`'s region of `input`, reduced into `out` and `indices` at
/// the same region. Each lane accumulates every `lanes`-th line of the plane's window, consecutive
/// lanes reading consecutive lines, the lanes meet across the plane
/// ([`Carrier::across`](cubecl::tile::Carrier::across)) and the first lane writes the total.
#[cube]
pub(crate) fn plane_line_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneLineReduceMicrokernel,
) {
    let item = microkernel.item();
    let rows = input.at(plane);
    let count = rows.flat::<In, N>().shape();
    let acc = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        &rows,
        UNIT_POS_PLANE as usize,
        count,
        0,
        comptime!(UnitStridedReduceMicrokernel::new(item, microkernel.lanes())),
    );
    let total = C::across(acc, UnitShare::Plane, item.operation.monoid());

    let load = rows.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let mut rows_out = out.at(plane);
    let rows_indices = indices.clone().map(|tile| tile.at(plane));
    if UNIT_POS_PLANE == 0 {
        unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
            total,
            &mut rows_out,
            &rows_indices,
            count * row_lanes,
            comptime!(UnitCarrierWriteMicrokernel::new(item, row_lanes)),
        );
    }
}
