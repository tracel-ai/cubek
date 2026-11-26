use crate::reduce::plane::PlaneShuffleReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LANE_SLOT_COUNT, LaunchedRows, MicrokernelLauncher, a_plane_a_cube,
    plane_lanes,
};
use crate::reduce::{ReduceCarrierFamily, ReduceWorkItem};
use cubecl::tile::Levels;
use cubecl::tile::launch::Decline;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A plane reduces its rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneShuffleMicrokernelLauncher;

impl MicrokernelLauncher for PlaneShuffleMicrokernelLauncher {
    fn levels(&self, hw: &Hardware, rows: &LaunchedRows) -> Levels {
        a_plane_a_cube(of_rows(hw, rows).levels(&rows.owner()))
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        plane_lanes(hw, item)
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        let microkernel = of_rows(&Hardware::from_client(launch.client), &launch.rows);
        launch_component!(
            launch,
            plane_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            microkernel,
            elements,
        );
    }
}

/// The plane microkernel of `rows` on `hw`.
fn of_rows(hw: &Hardware, rows: &LaunchedRows) -> PlaneShuffleReduceMicrokernel {
    PlaneShuffleReduceMicrokernel::new(rows.item, rows.plane_lane_count, LANE_SLOT_COUNT)
        .across_rows(hw, rows.line_lane_count, rows.row_count)
}

region_reduce_kernel!(plane_reduce_kernel, PlaneShuffleReduceMicrokernel);
