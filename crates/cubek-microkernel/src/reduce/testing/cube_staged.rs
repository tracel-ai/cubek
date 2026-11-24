use crate::reduce::cube::CubeStagedReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LANE_SLOT_COUNT, LaunchedRows, MicrokernelLauncher, cube_lanes,
};
use crate::reduce::{BATCH, ReduceCarrierFamily, ReduceWorkItem};
use cubecl::tile::Levels;
use cubecl::tile::launch::Decline;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A cube reduces its rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubeStagedMicrokernelLauncher {
    pub(crate) planes: usize,
}

impl MicrokernelLauncher for CubeStagedMicrokernelLauncher {
    fn levels(&self, hw: &Hardware, rows: &LaunchedRows) -> Levels {
        of_rows(hw, rows, self.planes)
            .levels(&rows.owner())
            .cubes(&[BATCH])
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        cube_lanes(hw, item, self.planes)
    }

    fn launch_plane_width(&self, hw: &Hardware, rows: &LaunchedRows) -> usize {
        of_rows(hw, rows, self.planes).launch_plane_width()
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        let hw = Hardware::from_client(launch.client);
        let microkernel = of_rows(&hw, &launch.rows, self.planes);
        launch_component!(
            launch,
            cube_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            microkernel,
            elements,
        );
    }
}

/// The cube microkernel of `planes` planes reducing `rows` on `hw`.
fn of_rows(hw: &Hardware, rows: &LaunchedRows, planes: usize) -> CubeStagedReduceMicrokernel {
    CubeStagedReduceMicrokernel::new(rows.item, rows.plane_lane_count, planes, LANE_SLOT_COUNT)
        .across_rows(hw, rows.line_lane_count, rows.row_count)
}

region_reduce_kernel!(cube_reduce_kernel, CubeStagedReduceMicrokernel);
