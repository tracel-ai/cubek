use super::microkernel::{unit_carrier_write_microkernel, unit_walk_write_microkernel};
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit writing what its carrier keeps to its window of the output: each slot's line, the
/// values divided by the row's length for a mean, the indices beside them where its item writes
/// both. Where `row_lanes` lanes of a line hold one row, the lanes are combined first and the row
/// written as one cell a slot; where each lane is its own row's, the lines are written as they are.
///
/// It is part of every reduce microkernel, which hands it the carrier its scope met and the window
/// of its owner's rows, and has no launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitCarrierWriteMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The lanes of a line that hold one row's values: the line's width along the row, one
    /// across rows.
    #[cube(comptime)]
    row_lanes: usize,
}

impl UnitCarrierWriteMicrokernel {
    /// A unit writing `item`'s carrier, `row_lanes` lanes of each line holding one row.
    pub fn new(item: ReduceWorkItem, row_lanes: usize) -> Self {
        Self { item, row_lanes }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn row_lanes(&self) -> usize {
        self.row_lanes
    }
}

#[cube]
impl UnitCarrierWriteMicrokernel {
    /// `acc` written to `out` and `indices`, `values` the values each of its rows held
    /// (`unit_carrier_write_microkernel`).
    ///
    /// # Panics
    ///
    /// Where its item writes the indices beside the values and `indices` is none, or writes
    /// indices its carrier does not keep.
    pub fn execute<
        Out: Numeric,
        Idx: Numeric,
        Acc: Numeric,
        N: Size,
        C: ReduceAccumulator<Acc, N>,
    >(
        &self,
        acc: C,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
            acc,
            out,
            indices,
            values,
            comptime!(UnitCarrierWriteMicrokernel {
                item: self.item,
                row_lanes: self.row_lanes
            }),
        )
    }
}

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
    /// As [`UnitCarrierWriteMicrokernel::execute`].
    pub fn execute<
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
