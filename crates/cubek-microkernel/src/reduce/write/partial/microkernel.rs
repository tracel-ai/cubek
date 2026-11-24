use super::base::UnitPartialStoreMicrokernel;
use crate::reduce::{ReduceAccumulator, takes_partial_indices};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// A partial is its slots' lines side by side.
#[cube]
pub(crate) fn unit_partial_store_microkernel<
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    acc: C,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    part: usize,
    #[comptime] microkernel: UnitPartialStoreMicrokernel,
) {
    let item = microkernel.item();
    let slots = item.operation.written_per_row();
    let handed = partial_indices.is_some();
    comptime!(assert!(
        takes_partial_indices(item.operation.carrier(), handed),
        "UnitPartialStoreMicrokernel: {:?} takes its partials' indices exactly where it keeps any",
        item.operation.carrier()
    ));

    #[unroll]
    for slot in 0..slots {
        let line = part * slots + slot;
        partials
            .flat_mut::<Acc, N>()
            .write(line, C::value_line(&acc, slot));
        #[comptime]
        match partial_indices.clone() {
            ComptimeOption::Some(mut tile) => tile
                .flat_mut::<u32, N>()
                .write(line, C::index_line(&acc, slot)),
            ComptimeOption::None => {}
        }
    }
}
