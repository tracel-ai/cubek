use super::base::UnitWalkReduceMicrokernel;
use crate::reduce::read::accumulate_line;
use crate::reduce::{REDUCED, ReduceAccumulator};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every step but the last reads without a check, as
/// [`unit_strided_reduce_microkernel`](crate::reduce::read::unit_strided_reduce_microkernel)'s do.
#[cube]
pub(crate) fn unit_walk_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    region: &Region,
    input: &Tile,
    values: usize,
    #[comptime] microkernel: UnitWalkReduceMicrokernel,
) -> C {
    let item = microkernel.item();
    let walk = region.walk();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let full_steps = values / microkernel.per_step();
    let mut acc = C::identity(item.operation.monoid());
    let first_cell = 0usize;
    for step in 0..full_steps {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            accumulate_line::<In, Acc, N, C>(
                &mut acc,
                &cell.flat::<In, N>(),
                first_cell,
                unit.origin(REDUCED),
                row_lanes,
                item,
            );
        }
    }

    // The step the rows' end cuts short, where one is: each unit checks its own position.
    for step in full_steps..values.div_ceil(microkernel.per_step()) {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            let position = unit.origin(REDUCED);
            if position < values {
                accumulate_line::<In, Acc, N, C>(
                    &mut acc,
                    &cell.flat::<In, N>(),
                    first_cell,
                    position,
                    row_lanes,
                    item,
                );
            }
        }
    }
    acc
}
