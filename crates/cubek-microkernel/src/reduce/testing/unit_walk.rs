use crate::reduce::plane::PlaneColumnsReduceMicrokernel;
use crate::reduce::read::UnitWalkReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
};
use crate::reduce::write::{UnitGroupColumnWriteMicrokernel, unit_group_column_write_microkernel};
use crate::reduce::{ReduceCarrierFamily, ReduceWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A plane's units lie across several lines of rows, and each column's first unit's carrier is written.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitWalkMicrokernelLauncher {
    pub(crate) columns: usize,
}

impl MicrokernelLauncher for UnitWalkMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        a_plane_a_cube(
            PlaneColumnsReduceMicrokernel::new(rows.item, rows.plane_lane_count, self.columns)
                .levels(&rows.owner()),
        )
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        plane_lanes(hw, item)
    }

    fn stride_on(&self, hw: &Hardware) -> usize {
        hw.plane_dim().unwrap_or(1) as usize / self.columns
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            unit_walk_reduce_kernel,
            sizes: [],
            [launch.rows.item],
            UnitWalkReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count / self.columns, self.columns),
            elements,
        );
    }
}

/// Every plane's rows, its units across lines of them, read by the walked read into a carrier of values alone, written by the column's write.
///
/// Its operands' leaves:
/// - `input`: the unit walk's rows, a row a lane, across the reduced dimension.
/// - `out`: the unit walk's cells, one a row.
#[cube(launch_unchecked)]
fn unit_walk_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] item: ReduceWorkItem,
    #[comptime] microkernel: UnitWalkReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for plane in &partitioning {
        let read = microkernel
            .runtime()
            .reduce::<In, Acc, N, Vector<Acc, N>>(&plane, &x);
        unit_group_column_write_microkernel::<Out, Out, Acc, N, Vector<Acc, N>>(
            &plane,
            read.carrier,
            &o,
            &ComptimeOption::new_None(),
            read.values,
            comptime!(UnitGroupColumnWriteMicrokernel::new(item)),
        );
    }
}
