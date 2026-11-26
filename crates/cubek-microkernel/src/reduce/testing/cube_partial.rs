use crate::reduce::cube::{CubePartialReduceMicrokernel, CubeStridedReduceMicrokernel};
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher, cube_lanes};
use crate::reduce::{BATCH, ReduceCarrierFamily, ReduceWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A cube reduces its rows into their one partial.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubePartialMicrokernelLauncher {
    planes: usize,
}

impl CubePartialMicrokernelLauncher {
    pub(crate) const fn new(planes: usize) -> Self {
        Self { planes }
    }
}

impl MicrokernelLauncher for CubePartialMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        rows.whole_rows().cubes(&[BATCH])
    }

    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        cube_lanes(hw, item, self.planes)
    }

    fn launch_plane_width(&self, _hw: &Hardware, rows: &LaunchedRows) -> usize {
        rows.plane_lane_count * self.planes
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let [input, accumulator, _] = launch.elements();
        let elements = [input, accumulator];
        launch_component!(
            launch,
            cube_partial_reduce_kernel,
            sizes: [],
            [],
            CubePartialReduceMicrokernel::new(CubeStridedReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count, self.planes)),
            elements,
        );
    }
}

/// Every cube's row, reduced by the cube into a carrier of values alone and stored to `out` as its
/// one partial.
///
/// Its operands' leaves:
/// - `input`: the rows, one value a line along the reduced dimension, or a row a lane across it.
/// - `out`: the partials, one line a row.
#[cube(launch_unchecked)]
fn cube_partial_reduce_kernel<In: Numeric, Acc: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubePartialReduceMicrokernel,
    #[define(In, Acc)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        let mut partials = o.at(&cube);
        let first = 0usize;
        microkernel.runtime().reduce::<In, Acc, N, Vector<Acc, N>>(
            &x.at(&cube),
            &mut partials,
            &ComptimeOption::new_None(),
            first,
            first,
        );
    }
}
