use super::microkernel::cube_columns_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
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
    /// Units of a plane, as wide as the device commits to.
    #[cube(comptime)]
    lanes: usize,
    /// Planes of the cube.
    #[cube(comptime)]
    planes: usize,
    /// Lines of rows its units lie across side by side, a power of two.
    #[cube(comptime)]
    columns: usize,
}

impl CubeColumnsReduceMicrokernel {
    /// A cube of `planes` planes of `lanes` units reducing `item`, each plane across `columns` lines of
    /// rows.
    pub fn new(item: ReduceWorkItem, lanes: usize, planes: usize, columns: usize) -> Self {
        Self {
            item,
            lanes,
            planes,
            columns,
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

    pub fn columns(&self) -> usize {
        self.columns
    }
}

#[cube]
impl CubeColumnsReduceMicrokernel {
    /// `cube`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`cube_columns_reduce_microkernel`).
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
