use crate::reduce::cube::across_planes;
use crate::reduce::read::{UnitWalkReduceMicrokernel, unit_walk_reduce_microkernel};
use crate::reduce::write::{UnitGroupColumnWriteMicrokernel, unit_group_column_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem, cube_columns, spreads_across};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Dimension, Levels, Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Rows reduced by one cube, read several lines of rows wide: a column of each plane's units a line,
/// the column's units of every plane taking the reduced dimension in turns, meeting within their
/// plane, then across planes in shared memory.
///
/// [`CubeStagedReduceMicrokernel`](crate::reduce::cube::CubeStagedReduceMicrokernel) runs it where
/// its planes lie across several lines.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeColumnsReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    planes: usize,
    #[cube(comptime)]
    columns: usize,
}

impl CubeColumnsReduceMicrokernel {
    /// # Panics
    ///
    /// Where `columns` is not a power of two at most one plane's `lanes`.
    pub fn new(item: ReduceWorkItem, lanes: usize, planes: usize, columns: usize) -> Self {
        assert!(
            spreads_across(lanes, columns),
            "CubeColumnsReduceMicrokernel: {columns} columns of {lanes} lanes"
        );
        Self {
            item,
            lanes,
            planes,
            columns,
        }
    }

    /// The levels below its cube from `owner`, as
    /// [`PlaneColumnsReduceMicrokernel::levels`](crate::reduce::plane::PlaneColumnsReduceMicrokernel::levels)
    /// of each plane.
    pub fn levels(&self, owner: &[(Dimension, usize)]) -> Levels {
        cube_columns(owner, self.columns, self.lanes, self.planes)
    }
}

#[cube]
impl CubeColumnsReduceMicrokernel {
    /// As [`CubeStagedReduceMicrokernel::reduce`](crate::reduce::cube::CubeStagedReduceMicrokernel::reduce),
    /// several lines of rows wide (`cube_columns_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where the cube's columns' item and `indices` disagree on whether the indices are written beside the values, or
    /// its carrier keeps none of the indices it writes.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: Accumulable<Acc, N> + Clone,
    >(
        &self,
        cube: &Region,
        input: &Tile,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        cube_columns_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            cube,
            input,
            out,
            indices,
            comptime!(CubeColumnsReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes,
                columns: self.columns
            }),
        )
    }
}

/// A plane's part of a column meets across its lanes' low bits, as
/// [`plane_columns_reduce_microkernel`](crate::reduce::plane::plane_columns_reduce_microkernel())'s
/// does; its unit at the run's start stages the partial, and the planes' columns meet in shared memory
/// ([`across_planes`](crate::reduce::cube::across_planes())).
#[cube]
pub(crate) fn cube_columns_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: Accumulable<Acc, N> + Clone,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeColumnsReduceMicrokernel,
) {
    let item = microkernel.item;
    let columns = microkernel.columns;
    let per_column = microkernel.lanes / columns;
    let per_step = microkernel.lanes * microkernel.planes / columns;
    let read = unit_walk_reduce_microkernel::<In, Acc, N, C>(
        cube,
        input,
        comptime!(UnitWalkReduceMicrokernel::new(item, per_step, columns)),
    );
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let plane = C::across(read.carrier, column, item.operation.monoid());
    let lane = UNIT_POS_PLANE as usize;
    let total = across_planes::<Acc, N, C>(
        plane,
        lane / per_column,
        lane.is_multiple_of(per_column),
        columns,
        microkernel.planes,
        item,
    );
    unit_group_column_write_microkernel::<Out, Idx, Acc, N, C>(
        cube,
        total,
        out,
        indices,
        read.values,
        comptime!(UnitGroupColumnWriteMicrokernel::new(item)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;
    use crate::reduce::testing::CubeColumnsMicrokernelLauncher;

    const LANE_COUNT: usize = 32;
    const PLANE_COUNT: usize = 4;

    #[test]
    #[should_panic(expected = "64 columns of 32 lanes")]
    fn more_columns_than_lanes_panic() {
        CubeColumnsReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_COUNT,
            PLANE_COUNT,
            2 * LANE_COUNT,
        );
    }

    /// The cube's read two lines of rows wide, 300 values long across the reduced dimension, each lane
    /// its own row.
    mod columns_across_300_long_four_wide {
        use super::*;
        every_operation!(CubeColumnsMicrokernelLauncher::new(4, 2), Across, 300, 4);
    }
}
