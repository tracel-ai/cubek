use super::base::UnitRegionWriteMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

#[cube]
pub(crate) fn unit_region_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    region: &Region,
    total: C,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    writes: bool,
    #[comptime] microkernel: UnitRegionWriteMicrokernel,
) {
    let mut cells = out.at(region);
    let cell_indices = indices.clone().map(|tile| tile.at(region));
    if writes {
        unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
            total,
            &mut cells,
            &cell_indices,
            values,
            comptime!(UnitCarrierWriteMicrokernel::new(
                microkernel.item(),
                microkernel.row_lanes()
            )),
        );
    }
}
