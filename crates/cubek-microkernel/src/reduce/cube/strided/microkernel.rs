use super::base::CubeStridedReduceMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::cube::across_planes;
use crate::reduce::read::{UnitStridedReduceMicrokernel, unit_strided_reduce_microkernel};
use cubecl::tile::Tile;
use cubecl::tile::space::UnitShare;
use cubecl::{self as cubecl, cube, prelude::*};

/// Consecutive units read consecutive lines, each plane's lanes meet in the plane's instructions and
/// the planes in shared memory ([`across_planes`](crate::reduce::cube::across_planes())).
#[cube]
pub(crate) fn cube_strided_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    first_line: usize,
    count: usize,
    #[comptime] microkernel: CubeStridedReduceMicrokernel,
) -> C {
    let monoid = microkernel.item().operation.monoid();
    let unit = PLANE_POS as usize * microkernel.lanes() + UNIT_POS_PLANE as usize;
    let acc = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        input,
        unit,
        count,
        first_line,
        comptime!(UnitStridedReduceMicrokernel::new(
            microkernel.item(),
            microkernel.lanes() * microkernel.planes()
        )),
    );

    let plane = C::across(acc, UnitShare::Plane, monoid);
    let one_column = comptime!(1usize);
    across_planes::<Acc, N, C>(
        plane,
        0,
        UNIT_POS_PLANE == 0,
        one_column,
        microkernel.planes(),
        microkernel.item(),
    )
}
