use super::base::UnitReduceMicrokernel;
use crate::reduce::{Folded, ReduceLayout, read_mapped};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's row: every line of `input`, the unit's window of the rows it reduces, read in
/// `In`, mapped and folded in `Acc` into `C`, and the total written to `out`, its window of the
/// output, in `Out`.
#[cube]
pub fn unit_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    C: Folded<Acc, N>,
>(
    input: &Tile,
    out: &mut Tile,
    #[comptime] microkernel: UnitReduceMicrokernel,
) {
    let operation = comptime!(microkernel.item.operation);
    let monoid = comptime!(operation.monoid());
    let layout = comptime!(microkernel.item.layout);
    let lines = input.flat::<In, N>();
    let count = lines.shape();
    let mut acc = C::identity(monoid);
    for line in 0..count {
        let read = read_mapped::<Acc, N>(
            Vector::<Acc, N>::cast_from(lines.read(line)),
            comptime!(operation.read_map()),
        );
        acc = C::fold(acc, read, line, layout, monoid);
    }
    let values = match comptime!(layout) {
        ReduceLayout::Along => count * N::value(),
        ReduceLayout::Across => count,
    };
    C::write::<Out>(acc, out, values, layout, operation);
}
