use super::base::UnitPartialsReduceMicrokernel;
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{ReduceAccumulator, assert_partial_indices};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// The first `parts` partials of `partials` and `partial_indices`, each its carrier's lines side by
/// side, combined in order by one unit and written to `out` and `indices`, `values` the values each
/// of the rows holds.
#[cube]
pub(crate) fn unit_partials_reduce_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    partials: &Tile,
    partial_indices: &ComptimeOption<Tile>,
    parts: usize,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitPartialsReduceMicrokernel,
) {
    let item = microkernel.item();
    let monoid = item.operation.monoid();
    let slots = item.operation.written_per_row();
    let handed = partial_indices.is_some();
    comptime!(assert_partial_indices(
        "UnitPartialsReduceMicrokernel",
        item.operation.carrier(),
        handed
    ));

    let value_lines = partials.flat::<Acc, N>();
    let mut merged = C::identity(monoid);
    for part in 0..parts {
        let mut stored = C::identity(monoid);
        #[unroll]
        for slot in 0..slots {
            let line = part * slots + slot;
            #[comptime]
            let index = match partial_indices {
                ComptimeOption::Some(tile) => tile.flat::<u32, N>().read(line),
                ComptimeOption::None => Vector::new(0u32),
            };
            C::set_line(&mut stored, slot, value_lines.read(line), index);
        }
        merged = C::combine(merged, stored, monoid);
    }

    unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
        merged,
        out,
        indices,
        values,
        comptime!(UnitCarrierWriteMicrokernel::new(
            item,
            microkernel.row_lanes()
        )),
    );
}
