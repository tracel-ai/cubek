use super::microkernel::cube_strided_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Lines that one cube reads together, each unit every `units`-th, and meets into one carrier its
/// first unit is left with.
///
/// It is the read of [`CubeLineReduceMicrokernel`](crate::reduce::cube::CubeLineReduceMicrokernel),
/// whose rows it holds whole, and of
/// [`CubePartialReduceMicrokernel`](crate::reduce::cube::CubePartialReduceMicrokernel), whose part.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStridedReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    planes: usize,
}

impl CubeStridedReduceMicrokernel {
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
impl CubeStridedReduceMicrokernel {
    /// The first `count` lines of `input` accumulated by the cube into `C`, the line at `l` read
    /// as line `first_line + l` of its row (`cube_strided_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
        &self,
        input: &Tile,
        first_line: usize,
        count: usize,
    ) -> C {
        cube_strided_reduce_microkernel::<In, Acc, N, C>(
            input,
            first_line,
            count,
            comptime!(CubeStridedReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes
            }),
        )
    }
}
