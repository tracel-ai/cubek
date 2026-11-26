use crate::reduce::ReduceCarrierFamily;
use crate::reduce::testing::{ComponentLaunch, LANE_SLOT_COUNT, LaunchedRows, MicrokernelLauncher};
use crate::reduce::unit::UnitDirectReduceMicrokernel;
use cubecl::ir::ElemType;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// Every unit reduces its own rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnitDirectMicrokernelLauncher;

impl MicrokernelLauncher for UnitDirectMicrokernelLauncher {
    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        rows.unit_levels()
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let elements = launch.elements();
        launch_component!(
            launch,
            unit_reduce_kernel::<F>,
            sizes: [launch.rows.cells()],
            [],
            UnitDirectReduceMicrokernel::new(launch.rows.item, LANE_SLOT_COUNT),
            elements,
        );
    }
}

/// Every unit's row, reduced by the unit alone.
///
/// Its operands' leaves:
/// - `input`: the unit direct's rows, laid along or across the reduced dimension.
/// - `out`: the unit direct's cells, one a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn unit_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);

        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &row,
                &mut row_out,
                &ComptimeOption::new_None(),
            );
    }
}
