use crate::reduce::{Accumulable, REDUCED, ReduceWorkItem, accumulate_read};
use cubecl::tile::Tile;
use cubecl::tile::kind::Masked;
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
    /// The lines of `input` from `first` on, every `stride`-th up to `count`, accumulated from
    /// the identity, line `l` as line `offset + l` of its row (`unit_strided_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        input: &Tile,
        first: usize,
        count: usize,
        offset: usize,
    ) -> C {
        unit_strided_reduce_microkernel::<In, Acc, N, C>(
            input,
            first,
            count,
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
    count: usize,
    offset: usize,
    #[comptime] microkernel: UnitStridedReduceMicrokernel,
) -> C {
    let item = microkernel.item;
    let stride = microkernel.stride;
    let lines = input.flat::<In, N>();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let mut acc = C::identity(item.operation.monoid());
    let full_steps = count / stride;
    for step in 0..full_steps {
        let line = first + step * stride;
        accumulate_line::<In, Acc, N, C>(&mut acc, &lines, line, offset + line, row_lanes, item);
    }

    let last = first + full_steps * stride;
    if last < count {
        accumulate_line::<In, Acc, N, C>(&mut acc, &lines, last, offset + last, row_lanes, item);
    }
    acc
}

/// Line `line` of `lines` accumulated into `acc` as line `index` of its row, `row_lanes` of its lanes
/// holding one row's values.
#[cube]
pub(crate) fn accumulate_line<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    acc: &mut C,
    lines: &Masked<'_, Vector<In, N>, usize>,
    line: usize,
    index: usize,
    #[comptime] row_lanes: usize,
    #[comptime] item: ReduceWorkItem,
) {
    accumulate_read::<In, Acc, N, C>(acc, lines.read(line), index, row_lanes, item);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceCarrierFamily;
    use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher};
    use crate::reduce::write::{UnitLinesWriteMicrokernel, unit_lines_write_microkernel};
    use cubecl::ir::ElemType;
    use cubecl::tile::{Levels, Partitioning, TileTensor};
    use cubek_std::Hardware;

    /// Each unit reads its row's lines a stride apart into its carrier, which the lines' write writes.
    #[derive(Debug, Clone, Copy)]
    struct UnitStridedMicrokernelLauncher {
        stride: usize,
    }

    impl MicrokernelLauncher for UnitStridedMicrokernelLauncher {
        fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
            rows.unit_levels()
        }

        fn stride_on(&self, _hw: &Hardware) -> usize {
            self.stride
        }

        fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
            let elements = launch.elements();
            launch_component!(
                launch,
                unit_strided_reduce_kernel,
                sizes: [],
                [],
                UnitStridedReduceMicrokernel::new(launch.rows.item, self.stride),
                elements,
            );
        }
    }

    /// Every unit's row read by the strided read into a carrier of values alone, written by the lines' write.
    ///
    /// Its operands' leaves:
    /// - `input`: one value a line along the reduced dimension, or a row a lane across it.
    /// - `out`: one tensor of a cell a row.
    #[cube(launch_unchecked)]
    fn unit_strided_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
        input: &TileTensor<'_>,
        out: &mut TileTensor<'_>,
        partitioning: Partitioning,
        #[comptime] microkernel: UnitStridedReduceMicrokernel,
        #[define(In, Acc, Out)] _elements: [ElemType; 3],
    ) {
        let x = input.tile(&partitioning);
        let o = out.tile_mut(&partitioning);

        for unit in &partitioning {
            let mut row_out = o.at(&unit);
            let row = x.at(&unit);
            let count = row.flat::<In, N>().shape();
            let first = 0usize;
            let carrier = microkernel
                .runtime()
                .reduce::<In, Acc, N, Vector<Acc, N>>(&row, first, count, first);
            unit_lines_write_microkernel::<Out, Out, Acc, N, N, Vector<Acc, N>>(
                carrier,
                &mut row_out,
                &ComptimeOption::new_None(),
                count,
                comptime!(UnitLinesWriteMicrokernel::new(microkernel.item)),
            );
        }
    }

    /// Rows 37 values long along the reduced dimension, a value a line: twelve full steps, then the
    /// last line, 36, read past them.
    mod along_37_long_one_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitStridedMicrokernelLauncher { stride: 3 }, Along, 37, 1, Plain, [
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
        reduce_tests!(UnitStridedMicrokernelLauncher { stride: 3 }, Along, 36, 1, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }

    /// Rows 37 values long across the reduced dimension, each lane its own row.
    mod across_37_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitStridedMicrokernelLauncher { stride: 3 }, Across, 37, 4, Plain, [
            sum: ReduceOperation::Sum,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }
}
