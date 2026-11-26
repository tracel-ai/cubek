use crate::reduce::ReduceCarrierFamily;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher};
use crate::reduce::write::UnitRegionWriteMicrokernel;
use cubecl::ir::ElemType;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// Each unit takes its one line of the input as its carrier of values and writes it to its own region.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitRegionMicrokernelLauncher;

impl MicrokernelLauncher for UnitRegionMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        rows.unit_levels()
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let [_, accumulator, out] = launch.elements();
        let elements = [accumulator, out];
        launch_component!(
            launch,
            unit_region_write_kernel,
            sizes: [],
            [launch.rows.value_count],
            UnitRegionWriteMicrokernel::new(launch.rows.item, launch.rows.row_lanes()),
            elements,
        );
    }
}

/// Every unit's one line of `input`, taken as its carrier of values alone and written to its region of `out`.
///
/// Its operands' leaves:
/// - `input`: the unit region's line of each unit, the values of a row along the reduced dimension, or a value of a
///   row a lane across it.
/// - `out`: the unit region's cells, one a row.
#[cube(launch_unchecked)]
fn unit_region_write_kernel<Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitRegionWriteMicrokernel,
    #[define(Acc, Out)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let row = x.at(&unit);
        let carrier = row.flat::<Acc, N>().read(0);
        microkernel
            .runtime()
            .write::<Out, Out, Acc, N, Vector<Acc, N>>(
                &unit,
                carrier,
                &o,
                &ComptimeOption::new_None(),
                values,
                true,
            );
    }
}
