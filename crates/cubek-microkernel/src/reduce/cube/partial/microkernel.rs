use super::base::CubePartialReduceMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::cube::{CubeStridedReduceMicrokernel, cube_strided_reduce_microkernel};
use crate::reduce::write::{UnitPartialStoreMicrokernel, unit_partial_store_microkernel};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a row: the lines of `input`, the line at `l` read as line `first_line + l` of
/// the row, reduced by the cube, whose first unit stores the carrier as partial `part` of `partials`
/// and `partial_indices`. A part the row's end cuts short reads the edge monoid's identity past it,
/// which leaves the carrier as it is: every line of the window is read.
#[cube]
pub(crate) fn cube_partial_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    part: usize,
    first_line: usize,
    #[comptime] microkernel: CubePartialReduceMicrokernel,
) {
    let cube = microkernel.cube();
    let count = input.flat::<In, N>().shape();
    let total = cube_strided_reduce_microkernel::<In, Acc, N, C>(
        input,
        first_line,
        count,
        comptime!(CubeStridedReduceMicrokernel::new(
            cube.item(),
            cube.lanes(),
            cube.planes()
        )),
    );

    if UNIT_POS == 0 {
        unit_partial_store_microkernel::<Acc, N, C>(
            total,
            partials,
            partial_indices,
            part,
            comptime!(UnitPartialStoreMicrokernel::new(cube.item())),
        );
    }
}
