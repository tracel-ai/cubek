use super::microkernel::cube_line_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
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
    /// The cube's `planes` planes of `lanes` units, reducing `item` a line at a time.
    pub fn new(item: ReduceWorkItem, lanes: usize, planes: usize) -> Self {
        Self {
            item,
            lanes,
            planes,
        }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn planes(&self) -> usize {
        self.planes
    }
}

#[cube]
impl CubeLineReduceMicrokernel {
    /// As [`CubeStagedReduceMicrokernel::reduce`](crate::reduce::cube::CubeStagedReduceMicrokernel::reduce),
    /// one line of rows wide (`cube_line_reduce_microkernel`).
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
