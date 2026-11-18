use super::base::PlaneShuffleReduceMicrokernel;
use crate::reduce::{ReduceAccumulator, ReduceCarrier, read_mapped, row_values};
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
pub(crate) fn plane_shuffle_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    match microkernel.item().operation.carrier() {
        ReduceCarrier::Values => plane_shuffle_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::Indexed => plane_shuffle_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::TopK { k } => {
            plane_shuffle_reduce_top_k::<In, Acc, Out, Idx, N>(input, out, indices, microkernel, k)
        }
        ReduceCarrier::TopValues { k } => plane_shuffle_reduce_top_values::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            microkernel,
            k,
        ),
    }
}

#[cube]
fn plane_shuffle_reduce<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    let operation = microkernel.item().operation;
    let layout = microkernel.item().layout;
    let count = input.flat::<In, N>().shape();
    let total = plane_shuffle_reduce_part::<In, Acc, N, C>(input, count, microkernel);
    let values = row_values::<N>(count, layout);

    if UNIT_POS_PLANE == 0 {
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

/// The `count` lines of `input` accumulated by the plane into `C`: each lane takes every
/// `lanes`-th line, consecutive lanes reading consecutive lines, and the lanes meet across the
/// plane. Every lane is left with the total.
#[cube]
fn plane_shuffle_reduce_part<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    input: &Tile,
    count: usize,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) -> C {
    let operation = microkernel.item().operation;
    let monoid = operation.monoid();
    let layout = microkernel.item().layout;
    let lanes = microkernel.lanes();
    let lane = UNIT_POS_PLANE as usize;
    let lines = input.flat::<In, N>();

    let mut acc = C::identity(monoid);
    for step in 0..count.div_ceil(lanes) {
        let line = lane + step * lanes;
        if line < count {
            C::accumulate(
                &mut acc,
                read_mapped::<Acc, N>(
                    Vector::<Acc, N>::cast_from(lines.read(line)),
                    operation.read_map(),
                ),
                line,
                layout,
                monoid,
            );
        }
    }

    C::across(acc, UnitShare::Plane, monoid)
}

/// [`plane_shuffle_reduce`] into a top-k of `k` slots, `K` bound in the function that uses it.
#[cube]
fn plane_shuffle_reduce_top_k<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    plane_shuffle_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(input, out, indices, microkernel)
}

/// [`plane_shuffle_reduce`] into a top-k of `k` values without their indices, `K` bound in the
/// function that uses it.
#[cube]
fn plane_shuffle_reduce_top_values<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    plane_shuffle_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
        input,
        out,
        indices,
        microkernel,
    )
}
