use crate::reduce::cube::{CubeStridedReduceMicrokernel, cube_strided_reduce_microkernel};
use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Rows reduced by one cube, read one line of rows wide, the cube's first unit writing the total.
///
/// [`CubeStagedReduceMicrokernel`](crate::reduce::cube::CubeStagedReduceMicrokernel) runs it where
/// its planes lie across one line.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeLineReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    planes: usize,
}

impl CubeLineReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, lanes: usize, planes: usize) -> Self {
        Self {
            item,
            lanes,
            planes,
        }
    }
}

#[cube]
impl CubeLineReduceMicrokernel {
    /// As [`CubeStagedReduceMicrokernel::reduce`](crate::reduce::cube::CubeStagedReduceMicrokernel::reduce),
    /// one line of rows wide (`cube_line_reduce_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - Those of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem), of its item.
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where one of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s
    /// Assumptions is broken.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: Accumulable<Acc, N>,
    >(
        &self,
        cube: &Region,
        input: &Tile,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        cube_line_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            cube,
            input,
            out,
            indices,
            comptime!(CubeLineReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes
            }),
        )
    }
}

/// The cube reduces its window as
/// [`CubeStridedReduceMicrokernel`](crate::reduce::cube::CubeStridedReduceMicrokernel) does, and its
/// first unit writes the total.
#[cube]
pub(crate) fn cube_line_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeLineReduceMicrokernel,
) {
    let item = microkernel.item;
    let read = cube_strided_reduce_microkernel::<In, Acc, N, C>(
        &input.at(cube),
        0,
        comptime!(CubeStridedReduceMicrokernel::new(
            item,
            microkernel.lanes,
            microkernel.planes
        )),
    );
    unit_region_write_microkernel::<Out, Idx, Acc, N, C>(
        cube,
        read.carrier,
        out,
        indices,
        read.value_count,
        UNIT_POS == 0,
        comptime!(UnitRegionWriteMicrokernel::new(item, read.row_lane_count)),
    );
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::CubeLineMicrokernelLauncher;

    /// The cube's read one line of rows wide, 300 values long along the reduced dimension, four values a
    /// line.
    mod line_along_300_long_four_wide {
        use super::*;
        every_operation!(CubeLineMicrokernelLauncher::new(4), Along, 300, 4);
    }

    /// The cube's read one line of rows wide, 37 values long across the reduced dimension, each lane its
    /// own row.
    mod line_across_37_long_four_wide {
        use super::*;
        every_operation!(CubeLineMicrokernelLauncher::new(4), Across, 37, 4);
    }
}
