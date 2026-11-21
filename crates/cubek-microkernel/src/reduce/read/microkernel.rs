use super::base::{UnitStridedReduceMicrokernel, UnitWalkReduceMicrokernel};
use crate::reduce::{REDUCED, ReduceAccumulator, accumulate_read};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// The lines of `input` from `first` on, every `stride`-th up to `count`, accumulated from the
/// identity, line `l` as line `offset + l` of its row: one unit's share of a row its plane or its
/// cube reads together.
///
/// Every step but the last reads without a check: a check before each read puts a branch between
/// consecutive loads, and a unit then waits on each line before it asks for the next. `first`
/// must be below `stride`, so that one step past the full ones covers the rest.
#[cube]
pub(crate) fn unit_strided_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    first: usize,
    count: usize,
    offset: usize,
    #[comptime] microkernel: UnitStridedReduceMicrokernel,
) -> C {
    let item = microkernel.item();
    let stride = microkernel.stride();
    let lines = input.flat::<In, N>();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let mut acc = C::identity(item.operation.monoid());
    let full_steps = count / stride;
    for step in 0..full_steps {
        let line = first + step * stride;
        accumulate_read::<In, Acc, N, C>(
            &mut acc,
            lines.read(line),
            offset + line,
            row_lanes,
            item,
        );
    }

    let last = first + full_steps * stride;
    if last < count {
        accumulate_read::<In, Acc, N, C>(
            &mut acc,
            lines.read(last),
            offset + last,
            row_lanes,
            item,
        );
    }
    acc
}

/// The cells of `input` `region`'s walk gives this unit, `per_step` positions of the reduced
/// dimension a step, `values` long, accumulated from the identity, each as its position along the
/// reduced dimension: one unit's share of rows its plane or its cube reads with its units across
/// them. Every step but the last reads without a check, as
/// [`unit_strided_reduce_microkernel`]'s do.
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

    if full_steps * microkernel.per_step() < values {
        for unit in walk.region(full_steps).leaves() {
            if unit.origin(REDUCED) < values {
                let cell = input.at(&unit);
                let load = cell.vector_tile();
                let row_lanes = comptime!(load.extent_along(REDUCED));
                let line = cell.flat::<In, N>().read(0);
                accumulate_read::<In, Acc, N, C>(
                    &mut acc,
                    line,
                    unit.origin(REDUCED),
                    row_lanes,
                    item,
                );
            }
        }
    }
    acc
}
