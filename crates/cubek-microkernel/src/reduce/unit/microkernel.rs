use super::base::UnitDirectReduceMicrokernel;
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator, ReduceCarrier, accumulate_read};
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's row: every line of `input`, the unit's window of the rows it reduces, read in
/// `In`, mapped and accumulated in `Acc` into its carrier, and the total written to `out`, its window of the
/// output, in `Out`.
///
/// Its operation picks what the row accumulates into ([`ReduceCarrier`]).
#[cube]
pub(crate) fn unit_direct_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
) {
    match microkernel.item().operation.carrier() {
        ReduceCarrier::Values => unit_direct_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::Indexed => unit_direct_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::TopK { k } => {
            unit_direct_reduce_top_k::<In, Acc, Out, Idx, N>(input, out, indices, microkernel, k)
        }
        ReduceCarrier::TopValues { k } => unit_direct_reduce_top_values::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            microkernel,
            k,
        ),
    }
}

#[cube]
fn unit_direct_reduce<
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
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
) {
    let monoid = microkernel.item().operation.monoid();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));

    let lines = input.flat::<In, N>();
    let count = lines.shape();

    let mut acc = C::identity(monoid);
    for line in 0..count {
        accumulate_read::<In, Acc, N, C>(
            &mut acc,
            lines.read(line),
            line,
            row_lanes,
            microkernel.item(),
        );
    }

    unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
        acc,
        out,
        indices,
        count * row_lanes,
        comptime!(UnitCarrierWriteMicrokernel::new(
            microkernel.item(),
            row_lanes
        )),
    );
}

/// [`unit_direct_reduce`] into a top-k of `k` slots, `K` bound in the function that uses it.
#[cube]
fn unit_direct_reduce_top_k<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    unit_direct_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(input, out, indices, microkernel)
}

/// [`unit_direct_reduce`] into a top-k of `k` values without their indices, `K` bound in the
/// function that uses it.
#[cube]
fn unit_direct_reduce_top_values<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    unit_direct_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
        input,
        out,
        indices,
        microkernel,
    )
}
