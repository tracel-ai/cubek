use super::base::CubeReduceMicrokernel;
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
pub fn cube_reduce_microkernel<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &mut Tile,
    #[comptime] microkernel: CubeReduceMicrokernel,
) {
    match comptime!(microkernel.item.operation.carrier()) {
        ReduceCarrier::Values => {
            cube_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(input, out, indices, microkernel)
        }
        ReduceCarrier::Indexed => {
            cube_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(input, out, indices, microkernel)
        }
        ReduceCarrier::TopK { k } => {
            let size!(K) = k;
            cube_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(input, out, indices, microkernel)
        }
        ReduceCarrier::TopValues { k } => {
            let size!(K) = k;
            cube_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
                input,
                out,
                indices,
                microkernel,
            )
        }
    }
}

#[cube]
fn cube_reduce<
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
    #[comptime] microkernel: CubeReduceMicrokernel,
) {
    let operation = comptime!(microkernel.item.operation);
    let monoid = comptime!(operation.monoid());
    let layout = comptime!(microkernel.item.layout);
    let units = comptime!(microkernel.lanes * microkernel.planes);
    let unit = PLANE_POS as usize * comptime!(microkernel.lanes) + UNIT_POS_PLANE as usize;
    let lines = input.flat::<In, N>();
    let count = lines.shape();
    let mut acc = C::identity(monoid);
    for step in 0..count.div_ceil(units) {
        let line = unit + step * units;
        if line < count {
            let read = read_mapped::<Acc, N>(
                Vector::<Acc, N>::cast_from(lines.read(line)),
                comptime!(operation.read_map()),
            );
            C::accumulate(&mut acc, read, line, layout, monoid);
        }
    }
    let plane = C::across(acc, UnitShare::Plane, monoid);
    let total = C::across_planes(plane, comptime!(microkernel.planes), monoid);
    if unit == 0 {
        let values = match comptime!(layout) {
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
            comptime!(microkernel.item.writes()),
        );
    }
}
