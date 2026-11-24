use crate::reduce::read::{UnitWalkReduceMicrokernel, unit_walk_reduce_microkernel};
use crate::reduce::write::{UnitGroupColumnWriteMicrokernel, unit_group_column_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem, plane_columns, spreads_across};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Dimension, Levels, Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Rows reduced by one plane, read several lines of rows wide: a column of units a line, each
/// column's units taking the reduced dimension in turns and meeting within it.
///
/// [`PlaneShuffleReduceMicrokernel`](crate::reduce::plane::PlaneShuffleReduceMicrokernel) runs it
/// where its units lie across several lines.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneColumnsReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    columns: usize,
}

impl PlaneColumnsReduceMicrokernel {
    /// # Panics
    ///
    /// Where `columns` is not a power of two at most `lanes`.
    pub fn new(item: ReduceWorkItem, lanes: usize, columns: usize) -> Self {
        assert!(
            spreads_across(lanes, columns),
            "PlaneColumnsReduceMicrokernel: {columns} columns of {lanes} lanes"
        );
        Self {
            item,
            lanes,
            columns,
        }
    }

    /// The levels below its plane from `owner`, the box of rows one line wide a row's window holds.
    pub fn levels(&self, owner: &[(Dimension, usize)]) -> Levels {
        plane_columns(owner, self.columns, self.lanes)
    }
}

#[cube]
impl PlaneColumnsReduceMicrokernel {
    /// As [`PlaneShuffleReduceMicrokernel::reduce`](crate::reduce::plane::PlaneShuffleReduceMicrokernel::reduce),
    /// several lines of rows wide (`plane_columns_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where the plane's columns' item and `indices` disagree on whether the indices are written beside the values, or
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
        plane: &Region,
        input: &Tile,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        plane_columns_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneColumnsReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                columns: self.columns
            }),
        )
    }
}

/// Each column's units take its lines in turns as `plane`'s walk gives them, so one load reads whole
/// transactions, and meet within the column. They hold the low bits of the lane, as `plane`'s units
/// level lays the reduced dimension innermost, so a column meets across those bits alone.
#[cube]
pub(crate) fn plane_columns_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: Accumulable<Acc, N> + Clone,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneColumnsReduceMicrokernel,
) {
    let item = microkernel.item;
    let columns = microkernel.columns;
    let per_column = microkernel.lanes / columns;
    let read = unit_walk_reduce_microkernel::<In, Acc, N, C>(
        plane,
        input,
        comptime!(UnitWalkReduceMicrokernel::new(item, per_column, columns)),
    );
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let total = C::across(read.carrier, column, item.operation.monoid());
    unit_group_column_write_microkernel::<Out, Idx, Acc, N, C>(
        plane,
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
    use crate::reduce::testing::PlaneColumnsMicrokernelLauncher;

    const LANE_COUNT: usize = 32;

    #[test]
    #[should_panic(expected = "3 columns of 32 lanes")]
    fn columns_that_are_no_power_of_two_panic() {
        PlaneColumnsReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_COUNT,
            3,
        );
    }

    /// The plane's read two lines of rows wide, 300 values long across the reduced dimension, each lane
    /// its own row.
    mod columns_across_300_long_four_wide {
        use super::*;
        every_operation!(
            PlaneColumnsMicrokernelLauncher { columns: 2 },
            Across,
            300,
            4
        );
    }
}
