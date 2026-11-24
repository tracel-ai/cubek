use super::microkernel::unit_walk_write_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// The write that ends a read [`UnitWalkReduceMicrokernel`](crate::reduce::read::UnitWalkReduceMicrokernel)
/// shares: each column of rows' carrier written to its cell of the output by the column's unit at
/// the start of the reduced dimension in the walk's first step, one write a column, a row a lane.
///
/// It is part of the plane and cube microkernels, after their units of a column meet, and has no
/// launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWalkWriteMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
}

impl UnitWalkWriteMicrokernel {
    /// The units writing `item`'s columns.
    pub fn new(item: ReduceWorkItem) -> Self {
        Self { item }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }
}

#[cube]
impl UnitWalkWriteMicrokernel {
    /// `total`, a column of rows' carrier, written to `out` and `indices` at the cell of
    /// `region`'s walk its column's first unit holds, `values` the values each row held
    /// (`unit_walk_write_microkernel`).
    ///
    /// # Panics
    ///
    /// As [`UnitCarrierWriteMicrokernel::write`](crate::reduce::write::UnitCarrierWriteMicrokernel::write).
    pub fn write<
        Out: Numeric,
        Idx: Numeric,
        Acc: Numeric,
        N: Size,
        C: ReduceAccumulator<Acc, N> + Clone,
    >(
        &self,
        region: &Region,
        total: C,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_walk_write_microkernel::<Out, Idx, Acc, N, C>(
            region,
            total,
            out,
            indices,
            values,
            comptime!(UnitWalkWriteMicrokernel { item: self.item }),
        )
    }
}
