use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
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
    /// `region`'s walk its column's first unit holds, `values` the values each row held
    /// (`unit_group_column_write_microkernel`).
    ///
    /// # Panics
    ///
    /// Where the column's item and `indices` disagree on whether the indices are written beside the values, or
    /// its carrier keeps none of the indices it writes.
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
        let mut cell_out = out.at(&unit);
        let cell_indices = indices.clone().map(|tile| tile.at(&unit));
        if unit.origin(REDUCED) == 0 {
            unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
                total.clone(),
                &mut cell_out,
                &cell_indices,
                values,
                comptime!(UnitCarrierWriteMicrokernel::new(
                    microkernel.item,
                    a_row_a_lane
                )),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceCarrierFamily;
    use crate::reduce::plane::PlaneColumnsReduceMicrokernel;
    use crate::reduce::testing::{
        ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
    };
    use cubecl::ir::ElemType;
    use cubecl::tile::launch::Decline;
    use cubecl::tile::{Levels, Partitioning, TileTensor};
    use cubek_std::Hardware;

    /// A plane's units lie across several lines of rows, each taking its line of the walk's first step as its carrier.
    #[derive(Debug, Clone, Copy)]
    struct UnitGroupColumnMicrokernelLauncher {
        columns: usize,
    }

    impl MicrokernelLauncher for UnitGroupColumnMicrokernelLauncher {
        fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
            a_plane_a_cube(
                PlaneColumnsReduceMicrokernel::new(rows.item, rows.lanes, self.columns)
                    .levels(&rows.owner()),
            )
        }

        fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
            plane_lanes(hw, item)
        }

        fn values_of_len(&self, _len: usize) -> usize {
            1
        }

        fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
            let [_, accumulator, out] = launch.elements();
            let elements = [accumulator, out];
            launch_component!(
                launch,
                unit_group_column_write_kernel,
                sizes: [],
                [],
                UnitGroupColumnWriteMicrokernel::new(launch.rows.item),
                elements,
            );
        }
    }

    /// Every plane's units' lines of the walk's first step, each taken as its carrier of values alone, and each column's first unit's written to `out`.
    ///
    /// Its operands' leaves:
    /// - `input`: the rows, a row a lane, across the reduced dimension.
    /// - `out`: one tensor of a cell a row.
    #[cube(launch_unchecked)]
    fn unit_group_column_write_kernel<Acc: Numeric, Out: Numeric, N: Size>(
        input: &TileTensor<'_>,
        out: &mut TileTensor<'_>,
        partitioning: Partitioning,
        #[comptime] microkernel: UnitGroupColumnWriteMicrokernel,
        #[define(Acc, Out)] _elements: [ElemType; 2],
    ) {
        let x = input.tile(&partitioning);
        let o = out.tile_mut(&partitioning);

        for plane in &partitioning {
            let mut carrier = Vector::<Acc, N>::new(Acc::from_int(0));
            for unit in plane.walk().region(0).leaves() {
                let cell = x.at(&unit);
                carrier = cell.flat::<Acc, N>().read(0);
            }
            let one_value = 1usize;
            microkernel
                .runtime()
                .write::<Out, Out, Acc, N, Vector<Acc, N>>(
                    &plane,
                    carrier,
                    &o,
                    &ComptimeOption::new_None(),
                    one_value,
                );
        }
    }

    /// Each column's first unit's line of rows 300 values long, written as it is, a row a lane.
    mod walk_across_300_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitGroupColumnMicrokernelLauncher { columns: 2 }, Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
        ]);
    }
}
