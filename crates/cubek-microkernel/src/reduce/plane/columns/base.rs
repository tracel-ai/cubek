use super::microkernel::plane_columns_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem, plane_columns, spreads_across};
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
    /// A plane of `lanes` units reducing `item`, its units across `columns` lines of rows.
    ///
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

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn columns(&self) -> usize {
        self.columns
    }
}

#[cube]
impl PlaneColumnsReduceMicrokernel {
    /// As [`PlaneShuffleReduceMicrokernel::reduce`](crate::reduce::plane::PlaneShuffleReduceMicrokernel::reduce),
    /// its `columns` lines of rows wide (`plane_columns_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// As [`UnitCarrierWriteMicrokernel::write`](crate::reduce::write::UnitCarrierWriteMicrokernel::write).
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: ReduceAccumulator<Acc, N> + Clone,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;

    const LANES: usize = 32;

    #[test]
    #[should_panic(expected = "3 columns of 32 lanes")]
    fn columns_that_are_no_power_of_two_panic() {
        PlaneColumnsReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANES,
            3,
        );
    }
}
