use crate::reduce::read::{UnitStridedReduceMicrokernel, unit_strided_reduce_microkernel};
use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
use crate::reduce::{Accumulable, REDUCED, ReduceWorkItem};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Rows reduced by one plane, read one line of rows wide: its lanes read
/// the lines in turns and meet across the plane.
///
/// [`PlaneShuffleReduceMicrokernel`](crate::reduce::plane::PlaneShuffleReduceMicrokernel) runs it
/// where its units lie across one line.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneLineReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lanes: usize,
}

impl PlaneLineReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, lanes: usize) -> Self {
        Self { item, lanes }
    }
}

#[cube]
impl PlaneLineReduceMicrokernel {
    /// As [`PlaneShuffleReduceMicrokernel::reduce`](crate::reduce::plane::PlaneShuffleReduceMicrokernel::reduce),
    /// one line of rows wide (`plane_line_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where the plane's line's item and `indices` disagree on whether the indices are written beside the values, or
    /// its carrier keeps none of the indices it writes.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: Accumulable<Acc, N>,
    >(
        &self,
        plane: &Region,
        input: &Tile,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        plane_line_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneLineReduceMicrokernel {
                item: self.item,
                lanes: self.lanes
            }),
        )
    }
}

/// Each lane accumulates every `lanes`-th line of the plane's window, consecutive lanes reading
/// consecutive lines, the lanes meet across the plane
/// ([`Carrier::across`](cubecl::tile::Carrier::across)) and the first lane writes the total.
#[cube]
pub(crate) fn plane_line_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneLineReduceMicrokernel,
) {
    let item = microkernel.item;
    let rows = input.at(plane);
    let count = rows.flat::<In, N>().shape();
    let acc = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        &rows,
        UNIT_POS_PLANE as usize,
        count,
        0,
        comptime!(UnitStridedReduceMicrokernel::new(item, microkernel.lanes)),
    );
    let total = C::across(acc, UnitShare::Plane, item.operation.monoid());

    let load = rows.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    unit_region_write_microkernel::<Out, Idx, Acc, N, C>(
        plane,
        total,
        out,
        indices,
        count * row_lanes,
        UNIT_POS_PLANE == 0,
        comptime!(UnitRegionWriteMicrokernel::new(microkernel.item, row_lanes)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceCarrierFamily;
    use crate::reduce::testing::{
        ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
    };
    use cubecl::tile::Levels;
    use cubecl::tile::launch::Decline;
    use cubek_std::Hardware;

    /// A plane reads one line of rows wide.
    #[derive(Debug, Clone, Copy)]
    struct PlaneLineMicrokernelLauncher;

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
                PlaneLineReduceMicrokernel::new(launch.rows.item, launch.rows.lanes),
                elements,
            );
        }
    }

    region_reduce_kernel!(plane_line_reduce_kernel, PlaneLineReduceMicrokernel);

    /// The plane's read one line of rows wide, 300 values long along the reduced dimension, four values
    /// a line.
    mod line_along_300_long_four_wide {
        use super::*;
        every_operation!(PlaneLineMicrokernelLauncher, Along, 300, 4);
    }

    /// The plane's read one line of rows wide, 37 values long across the reduced dimension, each lane
    /// its own row.
    mod line_across_37_long_four_wide {
        use super::*;
        every_operation!(PlaneLineMicrokernelLauncher, Across, 37, 4);
    }
}
