use super::base::PlaneShuffleReduceMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::plane::{
    PlaneColumnsReduceMicrokernel, PlaneLineReduceMicrokernel, plane_columns_reduce_microkernel,
    plane_line_reduce_microkernel,
};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's rows, `plane`'s region of `input`, reduced into `out` and `indices` at the same
/// region: read one line of rows wide ([`PlaneLineReduceMicrokernel`]), or
/// [`lines_across`](PlaneShuffleReduceMicrokernel::lines_across) lines of rows wide
/// ([`PlaneColumnsReduceMicrokernel`]).
#[cube]
pub(crate) fn plane_shuffle_reduce_microkernel<
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
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    match microkernel.lines_across() == 1 {
        true => plane_line_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneLineReduceMicrokernel::new(
                microkernel.item(),
                microkernel.lanes()
            )),
        ),
        false => plane_columns_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneColumnsReduceMicrokernel::new(
                microkernel.item(),
                microkernel.lanes(),
                microkernel.lines_across()
            )),
        ),
    }
}
