use crate::reduce::read::RowsRead;
use crate::reduce::{Accumulable, REDUCED, ReduceWorkItem, accumulate_read};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit's share of rows its plane or its cube reads with its units across them: the cells
/// a region's walk gives it, accumulated into its carrier, each as its position along the reduced dimension.
///
/// [`UnitStridedReduceMicrokernel`](crate::reduce::read::UnitStridedReduceMicrokernel) is its twin
/// where the units read one line of rows.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWalkReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The positions of the reduced dimension one step of the walk covers.
    #[cube(comptime)]
    per_step: usize,
    /// The lines of rows the units lie across.
    #[cube(comptime)]
    columns: usize,
}

impl UnitWalkReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, per_step: usize, columns: usize) -> Self {
        Self {
            item,
            per_step,
            columns,
        }
    }
}

#[cube]
impl UnitWalkReduceMicrokernel {
    /// The cells of `input` `region`'s walk gives this unit, accumulated from the identity
    /// (`unit_walk_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        region: &Region,
        input: &Tile,
    ) -> RowsRead<C> {
        unit_walk_reduce_microkernel::<In, Acc, N, C>(
            region,
            input,
            comptime!(UnitWalkReduceMicrokernel {
                item: self.item,
                per_step: self.per_step,
                columns: self.columns
            }),
        )
    }
}

/// Every step but the last reads without a check, as
/// [`unit_strided_reduce_microkernel`](crate::reduce::read::unit_strided_reduce_microkernel)'s do.
#[cube]
pub(crate) fn unit_walk_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    region: &Region,
    input: &Tile,
    #[comptime] microkernel: UnitWalkReduceMicrokernel,
) -> RowsRead<C> {
    let item = microkernel.item;
    let walk = region.walk();
    let window = input.at(region);
    let values = window.flat::<In, N>().shape() / microkernel.columns;
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let full_steps = values / microkernel.per_step;
    let mut acc = C::identity(item.operation.monoid());
    let first_cell = 0usize;
    for step in 0..full_steps {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            let line = cell.flat::<In, N>().read(first_cell);
            let position = unit.origin(REDUCED);
            accumulate_read::<In, Acc, N, C>(&mut acc, line, position, first_cell, row_lanes, item);
        }
    }

    // The step the rows' end cuts short, where one is: each unit checks its own position.
    for step in full_steps..values.div_ceil(microkernel.per_step) {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            let lines = cell.flat::<In, N>();
            let position = unit.origin(REDUCED);
            if position < values {
                accumulate_read::<In, Acc, N, C>(
                    &mut acc,
                    lines.read(first_cell),
                    position,
                    first_cell,
                    row_lanes,
                    item,
                );
            }
        }
    }
    RowsRead::<C> {
        carrier: acc,
        values,
        row_lanes,
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitWalkMicrokernelLauncher;

    /// Walked across rows 300 values long, each lane its own row: the last step of the walk is cut
    /// short.
    mod walk_across_300_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitWalkMicrokernelLauncher { columns: 2 }, Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
            max_abs: ReduceOperation::MaxAbs,
            any: ReduceOperation::Any,
            all: ReduceOperation::All,
        ]);
    }

    /// Walked across rows 320 values long, no step of the walk cut short whatever the plane's width.
    mod walk_across_320_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitWalkMicrokernelLauncher { columns: 2 }, Across, 320, 4, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }
}
