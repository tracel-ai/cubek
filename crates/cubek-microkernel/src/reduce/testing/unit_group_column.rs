use crate::reduce::plane::PlaneColumnsReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
};
use crate::reduce::write::UnitGroupColumnWriteMicrokernel;
use crate::reduce::{ReduceCarrierFamily, ReduceWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A plane's units lie across several lines of rows, each taking its line of the walk's first step as its carrier.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitGroupColumnMicrokernelLauncher {
    columns: usize,
}

impl UnitGroupColumnMicrokernelLauncher {
    pub(crate) const fn new(columns: usize) -> Self {
        Self { columns }
    }
}

impl MicrokernelLauncher for UnitGroupColumnMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        a_plane_a_cube(
            PlaneColumnsReduceMicrokernel::new(rows.item, rows.plane_lane_count, self.columns)
                .levels(&rows.owner()),
        )
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        plane_lanes(hw, item)
    }

    fn values_of_value_count(&self, _len: usize) -> usize {
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
/// - `input`: the unit group column's rows, a row a lane, across the reduced dimension.
/// - `out`: the unit group column's cells, one a row.
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
