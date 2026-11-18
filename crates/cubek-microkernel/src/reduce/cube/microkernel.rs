use super::base::CubeStagedReduceMicrokernel;
use crate::reduce::{ReduceAccumulator, ReduceCarrier, ReduceLayout, read_mapped};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's row: each of its units accumulates every `units`-th line of `input`, the cube's window of
/// the rows it reduces, consecutive units reading consecutive lines; each plane's lanes meet in the
/// plane's instructions, the planes meet in shared memory ([`ReduceAccumulator::across_planes`]), and the
/// cube's first unit writes the total to `out`, the cube's window of the output.
///
/// Its operation picks what the row accumulates into ([`ReduceCarrier`]).
#[cube]
pub(crate) fn cube_staged_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &mut Tile,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    match microkernel.item().operation.carrier() {
        ReduceCarrier::Values => cube_staged_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::Indexed => cube_staged_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::TopK { k } => {
            let size!(K) = k;
            cube_staged_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(
                input,
                out,
                indices,
                microkernel,
            )
        }
        ReduceCarrier::TopValues { k } => {
            let size!(K) = k;
            cube_staged_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
                input,
                out,
                indices,
                microkernel,
            )
        }
    }
}

#[cube]
fn cube_staged_reduce<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &mut Tile,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    let operation = microkernel.item().operation;
    let layout = microkernel.item().layout;
    let count = input.flat::<In, N>().shape();

    let total = cube_staged_reduce_part::<In, Acc, N, C>(input, 0, count, microkernel);

    if PLANE_POS == 0 && UNIT_POS_PLANE == 0 {
        let values = match layout {
            ReduceLayout::Along => count * N::value(),
            ReduceLayout::Across => count,
        };
        C::write::<Out, Idx>(
            total,
            out,
            indices,
            values,
            layout,
            operation,
            microkernel.item().writes(),
        );
    }
}

/// The first `count` lines of `input`, the line at `l` read as line `first_line + l` of its row,
/// accumulated by the cube into `C`: each unit takes every `units`-th line, consecutive units
/// reading consecutive lines, each plane's lanes meet in the plane's instructions and the planes in
/// shared memory ([`ReduceAccumulator::across_planes`]). Every unit of the cube is left with the
/// total.
#[cube]
pub(crate) fn cube_staged_reduce_part<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    first_line: usize,
    count: usize,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) -> C {
    let operation = microkernel.item().operation;
    let monoid = operation.monoid();
    let layout = microkernel.item().layout;
    let units = microkernel.lanes() * microkernel.planes();

    let unit = PLANE_POS as usize * microkernel.lanes() + UNIT_POS_PLANE as usize;

    let lines = input.flat::<In, N>();

    let mut acc = C::identity(monoid);
    for step in 0..count.div_ceil(units) {
        let line = unit + step * units;
        if line < count {
            let read = read_mapped::<Acc, N>(
                Vector::<Acc, N>::cast_from(lines.read(line)),
                operation.read_map(),
            );
            C::accumulate(&mut acc, read, first_line + line, layout, monoid);
        }
    }

    let plane = C::across(acc, UnitShare::Plane, monoid);
    C::across_planes(plane, microkernel.planes(), monoid)
}
