use super::base::UnitCarrierWriteMicrokernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::write::{UnitLinesWriteMicrokernel, unit_lines_write_microkernel};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// `acc` written by one unit to `out`, its window of the output, and `indices` beside it, slot by
/// slot, `values` the values each of its rows held. Where a line's lanes hold one row
/// ([`row_lanes`](UnitCarrierWriteMicrokernel::row_lanes) above one), the lanes are combined into
/// the row's carrier first and each slot written as one cell; the whole line, a row a lane,
/// otherwise.
#[cube]
pub(crate) fn unit_carrier_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    acc: C,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitCarrierWriteMicrokernel,
) {
    let item = microkernel.item();
    match microkernel.row_lanes() > 1 {
        true => unit_lines_write_microkernel::<Out, Idx, Acc, N, Const<1>, C>(
            C::one_row(acc, item.operation.monoid()),
            out,
            indices,
            values,
            comptime!(UnitLinesWriteMicrokernel::new(item)),
        ),
        false => unit_lines_write_microkernel::<Out, Idx, Acc, N, N, C>(
            acc,
            out,
            indices,
            values,
            comptime!(UnitLinesWriteMicrokernel::new(item)),
        ),
    }
}
