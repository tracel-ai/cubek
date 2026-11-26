use crate::reduce::read::RowsRead;
use crate::reduce::{Accumulable, REDUCED, ReduceWorkItem, accumulate_read};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit's share of a row its plane or its cube reads together: its lines in turns from
/// its first on, accumulated into its carrier, each as its line of the row.
///
/// It is part of the plane and cube microkernels, which meet the units' carriers after it.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitStridedReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The units that read the row together, one line each in turn.
    #[cube(comptime)]
    stride: usize,
}

impl UnitStridedReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, stride: usize) -> Self {
        Self { item, stride }
    }
}

#[cube]
impl UnitStridedReduceMicrokernel {
    /// The lines of `input` from `first` on, a turn apart, accumulated from the identity, line `l` as
    /// line `offset + l` of its row (`unit_strided_reduce_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - `first` is below the stride, the units that read the row together: one step past the full
    ///   ones then covers the rest.
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        input: &Tile,
        first: usize,
        offset: usize,
    ) -> RowsRead<C> {
        unit_strided_reduce_microkernel::<In, Acc, N, C>(
            input,
            first,
            offset,
            comptime!(UnitStridedReduceMicrokernel {
                item: self.item,
                stride: self.stride
            }),
        )
    }
}

/// Every step but the last reads without a check: a check before each read puts a branch between
/// consecutive loads, and a unit then waits on each line before it asks for the next. `first`
/// must be below `stride`, so that one step past the full ones covers the rest.
#[cube]
pub(crate) fn unit_strided_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    input: &Tile,
    first: usize,
    offset: usize,
    #[comptime] microkernel: UnitStridedReduceMicrokernel,
) -> RowsRead<C> {
    let item = microkernel.item;
    let stride = microkernel.stride;
    let lines = input.flat::<In, N>();
    let count = lines.shape();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let mut acc = C::identity(item.operation.monoid());
    let full_steps = count / stride;
    for step in 0..full_steps {
        let line = first + step * stride;
        accumulate_read::<In, Acc, N, C>(&mut acc, lines.read(line), offset, line, row_lanes, item);
    }

    let last = first + full_steps * stride;
    if last < count {
        accumulate_read::<In, Acc, N, C>(&mut acc, lines.read(last), offset, last, row_lanes, item);
    }
    RowsRead::<C> {
        carrier: acc,
        value_count: count * row_lanes,
        row_lane_count: row_lanes,
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitStridedMicrokernelLauncher;

    /// Rows 37 values long along the reduced dimension, a value a line: twelve full steps, then the
    /// last line, 36, read past them.
    mod along_37_long_one_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitStridedMicrokernelLauncher::new(3), Along, 37, 1, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
            max_abs: ReduceOperation::MaxAbs,
            any: ReduceOperation::Any,
            all: ReduceOperation::All,
        ]);
    }

    /// Rows 36 values long along the reduced dimension: twelve full steps and no line past them.
    mod along_36_long_one_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitStridedMicrokernelLauncher::new(3), Along, 36, 1, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }

    /// Rows 37 values long across the reduced dimension, each lane its own row.
    mod across_37_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitStridedMicrokernelLauncher::new(3), Across, 37, 4, Plain, [
            sum: ReduceOperation::Sum,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }
}
