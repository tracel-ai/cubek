use crate::reduce::cube::CubeStridedReduceMicrokernel;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher, cube_lanes};
use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
use crate::reduce::{BATCH, ReduceCarrierFamily, ReduceWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// A cube reads its rows into a carrier, whose line its first unit writes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubeStridedMicrokernelLauncher {
    pub(crate) planes: usize,
}

impl MicrokernelLauncher for CubeStridedMicrokernelLauncher {
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
        let elements = launch.elements();
        launch_component!(
            launch,
            cube_strided_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [launch.rows.item],
            CubeStridedReduceMicrokernel::new(launch.rows.item, launch.rows.plane_lane_count, self.planes),
            elements,
        );
    }
}

/// Every cube's row, read by the cube into its carrier, which the region's write writes.
///
/// Its operands' leaves:
/// - `input`: the cube strided's rows, laid along or across the reduced dimension.
/// - `out`: the cube strided's cells, one a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn cube_strided_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] item: ReduceWorkItem,
    #[comptime] microkernel: CubeStridedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        let first_line = 0usize;
        let read = microkernel
            .runtime()
            .reduce::<In, Acc, N, F::Carrier<Acc, N, K>>(&x.at(&cube), first_line);
        unit_region_write_microkernel::<Out, Out, Acc, N, F::Carrier<Acc, N, K>>(
            &cube,
            read.carrier,
            &o,
            &ComptimeOption::new_None(),
            read.values,
            UNIT_POS == 0,
            comptime!(UnitRegionWriteMicrokernel::new(item, read.row_lanes)),
        );
    }
}
