use super::microkernel::plane_columns_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
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
    /// Units of the plane, as wide as the device commits to.
    #[cube(comptime)]
    lanes: usize,
    /// Lines of rows its units lie across side by side, a power of two.
    #[cube(comptime)]
    columns: usize,
}

impl PlaneColumnsReduceMicrokernel {
    /// A plane of `lanes` units reducing `item`, its units across `columns` lines of rows.
    pub fn new(item: ReduceWorkItem, lanes: usize, columns: usize) -> Self {
        Self {
            item,
            lanes,
            columns,
        }
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
    /// `plane`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`plane_columns_reduce_microkernel`).
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
