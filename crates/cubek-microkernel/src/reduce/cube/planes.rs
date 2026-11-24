//! How a cube's planes meet: each stages its partial in shared memory, and the first combines them.

use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::{self as cubecl, cube, prelude::*};

/// The partials of the cube's planes, each already met across its own lanes, met in shared
/// memory, one total for each of the `columns` lines of rows its units lie across: the unit of
/// each plane and column where `stages` is set stages its partial, `acc`, in that column's slot of
/// its plane, the cube meets, the first plane's units combine their column's slots, and the cube
/// meets again so the slots may be staged anew. Only the first plane's units are left with their
/// column's total; the others hold the identity.
#[cube]
pub(crate) fn across_planes<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    acc: C,
    column: usize,
    stages: bool,
    #[comptime] columns: usize,
    #[comptime] planes: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let operation = item.operation;
    // A partial is its carrier's lines, a slot each, the indices beside where it keeps any.
    let lines = planes * columns * operation.written_per_row();
    let index_lines = comptime!(match operation.carrier().carries_indices() {
        true => lines,
        false => 1,
    });
    let mut values = Shared::<[Vector<Acc, N>]>::new_slice(lines);
    let mut indices = Shared::<[Vector<u32, N>]>::new_slice(index_lines);
    if stages {
        stage_partial::<Acc, N, C>(
            &mut values,
            &mut indices,
            PLANE_POS as usize * columns + column,
            acc,
            item,
        );
    }

    sync_cube();

    let total =
        merged_by_first_plane::<Acc, N, C>(&values, &indices, column, columns, planes, item);

    sync_cube();
    total
}

/// `acc` staged as partial `partial` of `values` and `indices`, its slots' lines side by side.
#[cube]
fn stage_partial<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    partial: usize,
    acc: C,
    #[comptime] item: ReduceWorkItem,
) {
    let operation = item.operation;
    let slots = operation.written_per_row();
    let with_indices = operation.carrier().carries_indices();
    #[unroll]
    for slot in 0..slots {
        values[partial * slots + slot] = C::value_line(&acc, slot);
        if with_indices {
            indices[partial * slots + slot] = C::index_line(&acc, slot);
        }
    }
}

/// The partials of `column` the cube's planes staged in `values` and `indices`, `columns` a
/// plane, combined by the cube's first plane; every other unit is left with the identity.
#[cube]
fn merged_by_first_plane<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] columns: usize,
    #[comptime] planes: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let mut total = C::identity(item.operation.monoid());
    if PLANE_POS == 0 {
        total = merged_partials::<Acc, N, C>(values, indices, column, columns, planes, item);
    }
    total
}

/// The partials of `column` the cube's planes staged in `values` and `indices`, `columns` a
/// plane, combined. A carrier of one slot unrolls its loop over the planes; a top-k's stays a
/// loop, as each of its partials is itself `k` lines.
#[cube]
fn merged_partials<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] columns: usize,
    #[comptime] planes: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let operation = item.operation;
    let monoid = operation.monoid();
    let one_slot = operation.written_per_row() == 1;
    let mut total = C::identity(monoid);
    #[unroll(one_slot)]
    for plane in 0..planes {
        let partial = staged_partial::<Acc, N, C>(values, indices, plane * columns + column, item);
        total = C::combine(total, partial, monoid);
    }
    total
}

/// Partial `partial` of `values` and `indices` as [`stage_partial`] staged it.
#[cube]
fn staged_partial<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    partial: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let operation = item.operation;
    let slots = operation.written_per_row();
    let with_indices = operation.carrier().carries_indices();
    let mut staged = C::identity(operation.monoid());
    #[unroll]
    for slot in 0..slots {
        let index = match with_indices {
            true => indices[partial * slots + slot],
            false => Vector::new(0u32),
        };
        C::set_line(&mut staged, slot, values[partial * slots + slot], index);
    }
    staged
}
