use crate::reduce::ReduceCarrierFamily;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher};
use crate::reduce::write::UnitPartialStoreMicrokernel;
use cubecl::ir::ElemType;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// Each unit takes its one line of the input as its carrier and stores it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitPartialMicrokernelLauncher;

impl MicrokernelLauncher for UnitPartialMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        rows.unit_levels()
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let [_, accumulator, _] = launch.elements();
        launch_component!(
            launch,
            unit_partial_store_kernel,
            sizes: [],
            [],
            UnitPartialStoreMicrokernel::new(launch.rows.item),
            accumulator,
        );
    }
}

/// Every unit's one line of `input`, taken as its carrier of values alone and stored to `out` as its one partial.
///
/// Its operands' leaves:
/// - `input`: one line a unit.
/// - `out`: the partials, one line a unit.
#[cube(launch_unchecked)]
fn unit_partial_store_kernel<Acc: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitPartialStoreMicrokernel,
    #[define(Acc)] _element: ElemType,
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut partials = o.at(&unit);
        let row = x.at(&unit);
        microkernel.runtime().store::<Acc, N, Vector<Acc, N>>(
            row.flat::<Acc, N>().read(0),
            &mut partials,
            &ComptimeOption::new_None(),
            0,
        );
    }
}
