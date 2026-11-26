use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
use crate::reduce::{Accumulable, REDUCED, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// The write that ends a read [`UnitWalkReduceMicrokernel`](crate::reduce::read::UnitWalkReduceMicrokernel)
/// shares: each column of rows' carrier written to its cell of the output by the column's unit at
/// the start of the reduced dimension in the walk's first step, one write a column, a row a lane.
///
/// A column's units call it once they met: the write is the group's, by its first unit.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitGroupColumnWriteMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
}

impl UnitGroupColumnWriteMicrokernel {
    pub fn new(item: ReduceWorkItem) -> Self {
        Self { item }
    }
}

#[cube]
impl UnitGroupColumnWriteMicrokernel {
    /// `total`, a column of rows' carrier, written to `out` and `indices` at the cell of
    /// `region`'s walk its column's first unit holds, `values` the values each row holds
    /// (`unit_group_column_write_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - Those of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem), of its item.
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where one of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s
    /// Assumptions is broken.
    pub fn write<
        Out: Numeric,
        Idx: Numeric,
        Acc: Numeric,
        N: Size,
        C: Accumulable<Acc, N> + Clone,
    >(
        &self,
        region: &Region,
        total: C,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_group_column_write_microkernel::<Out, Idx, Acc, N, C>(
            region,
            total,
            out,
            indices,
            values,
            comptime!(UnitGroupColumnWriteMicrokernel { item: self.item }),
        )
    }
}

/// One write a column, a row a lane.
#[cube]
pub(crate) fn unit_group_column_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N> + Clone,
>(
    region: &Region,
    total: C,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitGroupColumnWriteMicrokernel,
) {
    let a_row_a_lane = comptime!(1usize);
    for unit in region.walk().region(0).leaves() {
        unit_region_write_microkernel::<Out, Idx, Acc, N, C>(
            &unit,
            total.clone(),
            out,
            indices,
            values,
            unit.origin(REDUCED) == 0,
            comptime!(UnitRegionWriteMicrokernel::new(
                microkernel.item,
                a_row_a_lane
            )),
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitGroupColumnMicrokernelLauncher;

    /// Each column's first unit's line of rows 300 values long, written as it is, a row a lane.
    mod walk_across_300_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitGroupColumnMicrokernelLauncher::new(2), Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
        ]);
    }
}
