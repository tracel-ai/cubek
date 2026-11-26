use crate::reduce::plane::PlaneLineReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
};
use crate::reduce::{ReduceCarrierFamily, ReduceWorkItem};
use cubecl::tile::Levels;
use cubecl::tile::launch::Decline;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A plane reads one line of rows wide.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneLineMicrokernelLauncher;

impl MicrokernelLauncher for PlaneLineMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        a_plane_a_cube(Levels::leaf(&rows.owner()))
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        plane_lanes(hw, item)
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            plane_line_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            PlaneLineReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count),
            elements,
        );
    }
}

region_reduce_kernel!(plane_line_reduce_kernel, PlaneLineReduceMicrokernel);
