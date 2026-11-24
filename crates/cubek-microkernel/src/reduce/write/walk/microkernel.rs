use super::base::UnitWalkWriteMicrokernel;
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// `total`, a column of rows' carrier, written to `out` and `indices` by the column's unit at the
/// start of the reduced dimension in `region`'s walk, at that unit's cell, one write a column, a
/// row a lane, `values` the values each row held.
#[cube]
pub(crate) fn unit_walk_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    region: &Region,
    total: C,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitWalkWriteMicrokernel,
) {
    let a_row_a_lane = comptime!(1usize);
    for unit in region.walk().region(0).leaves() {
        let mut cell_out = out.at(&unit);
        let cell_indices = indices.clone().map(|tile| tile.at(&unit));
        if unit.origin(REDUCED) == 0 {
            unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
                total.clone(),
                &mut cell_out,
                &cell_indices,
                values,
                comptime!(UnitCarrierWriteMicrokernel::new(
                    microkernel.item(),
                    a_row_a_lane
                )),
            );
        }
    }
}
