use crate::reduce::read::{UnitStridedReduceMicrokernel, unit_strided_reduce_microkernel};
use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem};
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
    /// # Assumptions
    ///
    /// - `indices` holds a tile exactly where the item writes the indices beside the values
    ///   ([`ReduceWrites::Both`](crate::reduce::ReduceWrites::Both)).
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where an Assumption, or one of
    /// [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s, is broken.
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
    let read = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        &input.at(plane),
        UNIT_POS_PLANE as usize,
        0,
        comptime!(UnitStridedReduceMicrokernel::new(item, microkernel.lanes)),
    );
    let total = C::across(read.carrier, UnitShare::Plane, item.operation.monoid());
    unit_region_write_microkernel::<Out, Idx, Acc, N, C>(
        plane,
        total,
        out,
        indices,
        read.value_count,
        UNIT_POS_PLANE == 0,
        comptime!(UnitRegionWriteMicrokernel::new(item, read.row_lane_count)),
    );
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::PlaneLineMicrokernelLauncher;

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
