use super::microkernel::cube_columns_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem, cube_columns, spreads_across};
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
    /// A cube of `planes` planes of `lanes` units reducing `item`, each plane across `columns` lines of
    /// rows.
    ///
    /// # Panics
    ///
    /// As [`PlaneColumnsReduceMicrokernel::new`](crate::reduce::plane::PlaneColumnsReduceMicrokernel::new).
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
    /// As [`CubeStagedReduceMicrokernel::reduce`](crate::reduce::cube::CubeStagedReduceMicrokernel::reduce),
    /// several lines of rows wide (`cube_columns_reduce_microkernel`).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;

    const LANES: usize = 32;
    const PLANES: usize = 4;

    #[test]
    #[should_panic(expected = "64 columns of 32 lanes")]
    fn more_columns_than_lanes_panic() {
        CubeColumnsReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANES,
            PLANES,
            2 * LANES,
        );
    }
}
