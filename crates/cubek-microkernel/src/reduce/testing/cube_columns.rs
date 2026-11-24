use crate::reduce::cube::CubeColumnsReduceMicrokernel;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher, cube_lanes};
use crate::reduce::{BATCH, ReduceCarrierFamily, ReduceWorkItem};
use cubecl::tile::Levels;
use cubecl::tile::launch::Decline;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A cube reads several lines of rows wide, a column of units of every plane each.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubeColumnsMicrokernelLauncher {
    pub(crate) planes: usize,
    pub(crate) columns: usize,
}

impl MicrokernelLauncher for CubeColumnsMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        CubeColumnsReduceMicrokernel::new(
            rows.item,
            rows.plane_lane_count,
            self.planes,
            self.columns,
        )
        .levels(&rows.owner())
        .cubes(&[BATCH])
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        cube_lanes(hw, item, self.planes)
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            cube_columns_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            CubeColumnsReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count, self.planes, self.columns),
            elements,
        );
    }
}

region_reduce_kernel!(cube_columns_reduce_kernel, CubeColumnsReduceMicrokernel);
