use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit of a region's owner writing the region's carrier to its cells of the output: the write
/// that ends a read [`UnitStridedReduceMicrokernel`](crate::reduce::read::UnitStridedReduceMicrokernel)
/// shares, after the units meet.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegionWriteMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    row_lanes: usize,
}

impl UnitRegionWriteMicrokernel {
    pub fn new(item: ReduceWorkItem, row_lanes: usize) -> Self {
        Self { item, row_lanes }
    }
}

#[cube]
impl UnitRegionWriteMicrokernel {
    /// `total` written to `region`'s cells of `out` and `indices` by the unit that `writes`, `values`
    /// the values each row holds (`unit_region_write_microkernel`).
    ///
    /// # Panics
    ///
    /// Where the region's item and `indices` disagree on whether the indices are written beside the values, or
    /// its carrier keeps none of the indices it writes.
    pub fn write<Out: Numeric, Idx: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        region: &Region,
        total: C,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
        writes: bool,
    ) {
        unit_region_write_microkernel::<Out, Idx, Acc, N, C>(
            region,
            total,
            out,
            indices,
            values,
            writes,
            comptime!(UnitRegionWriteMicrokernel {
                item: self.item,
                row_lanes: self.row_lanes
            }),
        )
    }
}

#[cube]
pub(crate) fn unit_region_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
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
                microkernel.item,
                microkernel.row_lanes
            )),
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitRegionMicrokernelLauncher;

    /// The write of a region's carrier: four values a line, met into the row's cell.
    mod region_along_one_line_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitRegionMicrokernelLauncher, Along, 4, 4, Plain, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
        ]);
    }

    /// The write of a region's carrier, a row a lane.
    mod region_across_one_value_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitRegionMicrokernelLauncher, Across, 1, 4, Plain, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
        ]);
    }
}
