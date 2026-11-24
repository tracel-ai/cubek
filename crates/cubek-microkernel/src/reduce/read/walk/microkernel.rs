use super::base::UnitWalkReduceMicrokernel;
use crate::reduce::{REDUCED, ReduceAccumulator, accumulate_read};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// The cells of `input` `region`'s walk gives this unit, `per_step` positions of the reduced
/// dimension a step, `values` long, accumulated from the identity, each as its position along the
/// reduced dimension: one unit's share of rows its plane or its cube reads with its units across
/// them. Every step but the last reads without a check, as
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
    let full_steps = values / microkernel.per_step();
    let mut acc = C::identity(item.operation.monoid());
    for step in 0..full_steps {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            let load = cell.vector_tile();
            let row_lanes = comptime!(load.extent_along(REDUCED));
            let line = cell.flat::<In, N>().read(0);
            accumulate_read::<In, Acc, N, C>(&mut acc, line, unit.origin(REDUCED), row_lanes, item);
        }
    }

    // The step the rows' end cuts short, where one is: each unit checks its own position.
    for step in full_steps..values.div_ceil(microkernel.per_step()) {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            let load = cell.vector_tile();
            let row_lanes = comptime!(load.extent_along(REDUCED));
            let position = unit.origin(REDUCED);
            if position < values {
                accumulate_read::<In, Acc, N, C>(
                    &mut acc,
                    cell.flat::<In, N>().read(0),
                    position,
                    row_lanes,
                    item,
                );
            }
        }
    }
    acc
}
