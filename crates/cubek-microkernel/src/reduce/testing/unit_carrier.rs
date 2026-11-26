use crate::reduce::ReduceCarrierFamily;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher};
use crate::reduce::write::UnitCarrierWriteMicrokernel;
use cubecl::ir::ElemType;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// Each unit takes its one line of the input as its carrier of values and writes it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitCarrierMicrokernelLauncher;

impl MicrokernelLauncher for UnitCarrierMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        rows.unit_levels()
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let [_, accumulator, out] = launch.elements();
        let elements = [accumulator, out];
        launch_component!(
            launch,
            unit_carrier_write_kernel,
            sizes: [],
            [launch.rows.value_count],
            UnitCarrierWriteMicrokernel::new(launch.rows.item, launch.rows.row_lanes()),
            elements,
        );
    }
}

/// Every unit's one line of `input`, taken as its carrier of values alone and written to `out`, a mean dividing by `values`.
///
/// Its operands' leaves:
/// - `input`: the unit carrier's line of each unit, the values of a row along the reduced dimension, or a value of a
///   row a lane across it.
/// - `out`: the unit carrier's cells, one a row.
#[cube(launch_unchecked)]
fn unit_carrier_write_kernel<Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitCarrierWriteMicrokernel,
    #[define(Acc, Out)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let row = x.at(&unit);
        let carrier = row.flat::<Acc, N>().read(0);
        let mut row_out = o.at(&unit);
        microkernel
            .runtime()
            .write::<Out, Out, Acc, N, Vector<Acc, N>>(
                carrier,
                &mut row_out,
                &ComptimeOption::new_None(),
                values,
            );
    }
}
