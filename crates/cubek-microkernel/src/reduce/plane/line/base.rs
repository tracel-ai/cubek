use super::microkernel::plane_line_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Rows reduced by one plane, read one line of rows wide: each lane reads every `lanes`-th line and
/// the lanes meet across the plane.
///
/// [`PlaneShuffleReduceMicrokernel`](crate::reduce::plane::PlaneShuffleReduceMicrokernel) runs it
/// where its units lie across one line.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneLineReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// Units of the plane, as wide as the device commits to.
    #[cube(comptime)]
    lanes: usize,
}

impl PlaneLineReduceMicrokernel {
    /// A plane of `lanes` units reducing `item`.
    pub fn new(item: ReduceWorkItem, lanes: usize) -> Self {
        Self { item, lanes }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
}

#[cube]
impl PlaneLineReduceMicrokernel {
    /// `plane`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`plane_line_reduce_microkernel`).
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
        C: ReduceAccumulator<Acc, N>,
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
