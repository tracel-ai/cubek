use crate::reduce::ReduceCarrierFamily;
use crate::reduce::read::UnitPartialsReduceMicrokernel;
use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher};
use cubecl::ir::ElemType;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// Each unit takes every line of its row as a partial of values alone.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitPartialsMicrokernelLauncher;

impl MicrokernelLauncher for UnitPartialsMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        rows.unit_levels()
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            unit_partials_reduce_kernel,
            sizes: [],
            [launch.rows.value_count],
            UnitPartialsReduceMicrokernel::new(launch.rows.item, launch.rows.row_lanes()),
            elements,
        );
    }
}

/// Every unit's row of `input`, each line taken as a partial of values alone, a mean dividing by `values`.
///
/// Its operands' leaves:
/// - `input`: the partials of each row, laid along or across the reduced dimension.
/// - `out`: the unit partials's cells, one a row.
#[cube(launch_unchecked)]
fn unit_partials_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitPartialsReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let partials = x.at(&unit);
        let parts = partials.flat::<In, N>().shape();
        microkernel
            .runtime()
            .reduce::<Out, Out, Acc, N, Vector<Acc, N>>(
                &partials,
                &ComptimeOption::new_None(),
                parts,
                &mut row_out,
                &ComptimeOption::new_None(),
                values,
            );
    }
}
