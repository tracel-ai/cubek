use super::base::UnitStridedReduceMicrokernel;
use crate::reduce::{REDUCED, ReduceAccumulator, accumulate_read};
use cubecl::tile::Tile;
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
