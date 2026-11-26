use super::base::CubeColumnsReduceMicrokernel;
use crate::reduce::cube::across_planes;
use crate::reduce::read::{UnitWalkReduceMicrokernel, unit_walk_reduce_microkernel};
use crate::reduce::write::{UnitWalkWriteMicrokernel, unit_walk_write_microkernel};
use crate::reduce::{ReduceAccumulator, values_of_columns};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// A plane's part of a column meets across its lanes' low bits, as
/// [`plane_columns_reduce_microkernel`](crate::reduce::plane::plane_columns_reduce_microkernel())'s
/// does; its unit at the run's start stages the partial, and the planes' columns meet in shared memory
/// ([`across_planes`](crate::reduce::cube::across_planes())).
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
    let values = values_of_columns::<Vector<In, N>>(&window.flat::<In, N>(), columns);
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
