use crate::reduce::cube::CubeLineReduceMicrokernel;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher, cube_lanes};
use crate::reduce::{BATCH, ReduceCarrierFamily, ReduceWorkItem};
use cubecl::tile::Levels;
use cubecl::tile::launch::Decline;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A cube reads one line of rows wide.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubeLineMicrokernelLauncher {
    planes: usize,
}

impl CubeLineMicrokernelLauncher {
    pub(crate) const fn new(planes: usize) -> Self {
        Self { planes }
    }
}

impl MicrokernelLauncher for CubeLineMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        Levels::leaf(&rows.owner()).cubes(&[BATCH])
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        cube_lanes(hw, item, self.planes)
    }

    fn launch_plane_width(&self, _hw: &Hardware, rows: &LaunchedRows) -> usize {
        rows.plane_lane_count * self.planes
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            cube_line_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            CubeLineReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count, self.planes),
            elements,
        );
    }
}

region_reduce_kernel!(cube_line_reduce_kernel, CubeLineReduceMicrokernel);
