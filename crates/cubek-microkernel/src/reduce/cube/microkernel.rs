use super::base::CubeStagedReduceMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::cube::{
    CubeColumnsReduceMicrokernel, CubeLineReduceMicrokernel, cube_columns_reduce_microkernel,
    cube_line_reduce_microkernel,
};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, `cube`'s region of `input`, reduced into `out` and `indices` at the same
/// region: read one line of rows wide ([`CubeLineReduceMicrokernel`]), or
/// [`lines_across`](CubeStagedReduceMicrokernel::lines_across) lines of rows wide
/// ([`CubeColumnsReduceMicrokernel`]).
#[cube]
pub(crate) fn cube_staged_reduce_microkernel<
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
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    match microkernel.lines_across() == 1 {
        true => cube_line_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            cube,
            input,
            out,
            indices,
            comptime!(CubeLineReduceMicrokernel::new(
                microkernel.item(),
                microkernel.lanes(),
                microkernel.planes()
            )),
        ),
        false => cube_columns_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            cube,
            input,
            out,
            indices,
            comptime!(CubeColumnsReduceMicrokernel::new(
                microkernel.item(),
                microkernel.lanes(),
                microkernel.planes(),
                microkernel.lines_across()
            )),
        ),
    }
}
