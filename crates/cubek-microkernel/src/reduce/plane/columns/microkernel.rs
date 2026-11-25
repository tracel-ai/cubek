use super::base::PlaneColumnsReduceMicrokernel;
use crate::reduce::read::{UnitWalkReduceMicrokernel, unit_walk_reduce_microkernel};
use crate::reduce::write::{UnitWalkWriteMicrokernel, unit_walk_write_microkernel};
use crate::reduce::{ReduceAccumulator, values_of_columns};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// Each column's units take its lines in turns as `plane`'s walk gives them, so one load reads whole
/// transactions, and meet within the column. They hold the low bits of the lane, as `plane`'s units
/// level lays the reduced dimension innermost, so a column meets across those bits alone.
#[cube]
pub(crate) fn plane_columns_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneColumnsReduceMicrokernel,
) {
    let item = microkernel.item();
    let columns = microkernel.columns();
    let per_column = microkernel.lanes() / columns;
    let window = input.at(plane);
    let values = values_of_columns::<Vector<In, N>>(&window.flat::<In, N>(), columns);
    let acc = unit_walk_reduce_microkernel::<In, Acc, N, C>(
        plane,
        input,
        values,
        comptime!(UnitWalkReduceMicrokernel::new(item, per_column)),
    );
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let total = C::across(acc, column, item.operation.monoid());
    unit_walk_write_microkernel::<Out, Idx, Acc, N, C>(
        plane,
        total,
        out,
        indices,
        values,
        comptime!(UnitWalkWriteMicrokernel::new(item)),
    );
}
