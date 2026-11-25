use super::microkernel::unit_region_write_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
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
    /// The unit writing `item`'s carrier, `row_lanes` lanes of each line holding one row.
    pub fn new(item: ReduceWorkItem, row_lanes: usize) -> Self {
        Self { item, row_lanes }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn row_lanes(&self) -> usize {
        self.row_lanes
    }
}

#[cube]
impl UnitRegionWriteMicrokernel {
    /// `total` written to `region`'s cells of `out` and `indices` by the unit that `writes`, `values`
    /// the values each row held (`unit_region_write_microkernel`).
    ///
    /// # Panics
    ///
    /// As [`UnitCarrierWriteMicrokernel::write`](crate::reduce::write::UnitCarrierWriteMicrokernel::write).
    pub fn write<
        Out: Numeric,
        Idx: Numeric,
        Acc: Numeric,
        N: Size,
        C: ReduceAccumulator<Acc, N>,
    >(
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
