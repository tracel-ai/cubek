use crate::reduce::plane::PlaneColumnsReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
};
use crate::reduce::{ReduceCarrierFamily, ReduceWorkItem};
use cubecl::tile::Levels;
use cubecl::tile::launch::Decline;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A plane reads several lines of rows wide, a column of units each.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneColumnsMicrokernelLauncher {
    columns: usize,
}

impl PlaneColumnsMicrokernelLauncher {
    pub(crate) const fn new(columns: usize) -> Self {
        Self { columns }
    }
}

impl MicrokernelLauncher for PlaneColumnsMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        a_plane_a_cube(
            PlaneColumnsReduceMicrokernel::new(rows.item, rows.plane_lane_count, self.columns)
                .levels(&rows.owner()),
        )
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        plane_lanes(hw, item)
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            plane_columns_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            PlaneColumnsReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count, self.columns),
            elements,
        );
    }
}

region_reduce_kernel!(plane_columns_reduce_kernel, PlaneColumnsReduceMicrokernel);
