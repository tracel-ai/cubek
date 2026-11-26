use crate::reduce::read::UnitStridedReduceMicrokernel;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher};
use crate::reduce::write::{UnitLinesWriteMicrokernel, unit_lines_write_microkernel};
use crate::reduce::{ReduceCarrierFamily, ReduceWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// Each unit reads its row's lines a stride apart into its carrier, which the lines' write writes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitStridedMicrokernelLauncher {
    stride: usize,
}

impl UnitStridedMicrokernelLauncher {
    pub(crate) const fn new(stride: usize) -> Self {
        Self { stride }
    }
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
            [launch.rows.item],
            UnitStridedReduceMicrokernel::new(launch.rows.item, self.stride),
            elements,
        );
    }
}

/// Every unit's row read by the strided read into a carrier of values alone, written by the lines' write.
///
/// Its operands' leaves:
/// - `input`: one value a line along the reduced dimension, or a row a lane across it.
/// - `out`: the unit strided's cells, one a row.
#[cube(launch_unchecked)]
fn unit_strided_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] item: ReduceWorkItem,
    #[comptime] microkernel: UnitStridedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);
        let first = 0usize;
        let read = microkernel
            .runtime()
            .reduce::<In, Acc, N, Vector<Acc, N>>(&row, first, first);
        unit_lines_write_microkernel::<Out, Out, Acc, N, N, Vector<Acc, N>>(
            read.carrier,
            &mut row_out,
            &ComptimeOption::new_None(),
            read.value_count,
            comptime!(UnitLinesWriteMicrokernel::new(item)),
        );
    }
}
