use super::base::CubeColumnsReduceMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::cube::across_planes;
use crate::reduce::read::{UnitWalkReduceMicrokernel, unit_walk_reduce_microkernel};
use crate::reduce::write::{UnitWalkWriteMicrokernel, unit_walk_write_microkernel};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, several lines of them wide: `cube`'s region of `input`, reduced into `out` and
/// `indices` at the same region. Each column's units, in every plane, take its lines in turns as
/// `cube`'s walk gives them, so one load reads whole transactions. They hold the low bits of their
/// plane's lane, as `cube`'s units level lays the reduced dimension innermost, so a plane's part of a
/// column meets across those bits alone; its unit at the run's start stages the partial, and the
/// planes' columns meet in shared memory ([`across_planes`](crate::reduce::cube::across_planes())).
#[cube]
pub(crate) fn cube_columns_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeColumnsReduceMicrokernel,
) {
    let item = microkernel.item();
    let columns = microkernel.columns();
    let per_column = microkernel.lanes() / columns;
    let per_step = microkernel.lanes() * microkernel.planes() / columns;
    let window = input.at(cube);
    let values = window.flat::<In, N>().shape() / columns;
    let acc = unit_walk_reduce_microkernel::<In, Acc, N, C>(
        cube,
        input,
        values,
        comptime!(UnitWalkReduceMicrokernel::new(item, per_step)),
    );
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let plane = C::across(acc, column, item.operation.monoid());
    let lane = UNIT_POS_PLANE as usize;
    let total = across_planes::<Acc, N, C>(
        plane,
        lane / per_column,
        lane.is_multiple_of(per_column),
        columns,
        microkernel.planes(),
        item,
    );
    unit_walk_write_microkernel::<Out, Idx, Acc, N, C>(
        cube,
        total,
        out,
        indices,
        values,
        comptime!(UnitWalkWriteMicrokernel::new(item)),
    );
}
