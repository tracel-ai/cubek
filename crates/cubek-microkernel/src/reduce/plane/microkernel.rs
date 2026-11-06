use super::base::PlaneReduceMicrokernel;
use crate::reduce::{ReduceAccumulator, ReduceCarrier, ReduceLayout, read_mapped};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's row: each lane accumulates every `lanes`-th line of `input`, the plane's window of the
/// rows it reduces, into its own carrier, consecutive lanes reading consecutive lines; the lanes'
/// partials meet across the plane ([`Carrier::across`](cubecl::tile::Carrier::across)), and the
/// first lane writes the total to `out`, the plane's window of the output.
///
/// Its operation picks what the row accumulates into ([`ReduceCarrier`]).
#[cube]
pub fn plane_reduce_microkernel<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &mut Tile,
    #[comptime] microkernel: PlaneReduceMicrokernel,
) {
    match comptime!(microkernel.item.operation.carrier()) {
        ReduceCarrier::Values => {
            plane_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(input, out, indices, microkernel)
        }
        ReduceCarrier::Indexed => {
            plane_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(input, out, indices, microkernel)
        }
        ReduceCarrier::TopK { k } => {
            let size!(K) = k;
            plane_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(input, out, indices, microkernel)
        }
        ReduceCarrier::TopValues { k } => {
            let size!(K) = k;
            plane_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
                input,
                out,
                indices,
                microkernel,
            )
        }
    }
}

#[cube]
fn plane_reduce<
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
    #[comptime] microkernel: PlaneReduceMicrokernel,
) {
    let operation = comptime!(microkernel.item.operation);
    let monoid = comptime!(operation.monoid());
    let layout = comptime!(microkernel.item.layout);
    let lanes = comptime!(microkernel.lanes);
    let lane = UNIT_POS_PLANE as usize;
    let lines = input.flat::<In, N>();
    let count = lines.shape();
    let mut acc = C::identity(monoid);
    for step in 0..count.div_ceil(lanes) {
        let line = lane + step * lanes;
        if line < count {
            let read = read_mapped::<Acc, N>(
                Vector::<Acc, N>::cast_from(lines.read(line)),
                comptime!(operation.read_map()),
            );
            C::accumulate(&mut acc, read, line, layout, monoid);
        }
    }
    let total = C::across(acc, UnitShare::Plane, monoid);
    if lane == 0 {
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
