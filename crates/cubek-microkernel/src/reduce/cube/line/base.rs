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
    /// Units of a plane, as wide as the device commits to.
    #[cube(comptime)]
    lanes: usize,
    /// Planes of the cube.
    #[cube(comptime)]
    planes: usize,
}

impl CubeLineReduceMicrokernel {
    /// A cube of `planes` planes of `lanes` units reducing `item`.
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
    /// `cube`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`cube_line_reduce_microkernel`).
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
