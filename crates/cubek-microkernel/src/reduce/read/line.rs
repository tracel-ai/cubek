use crate::reduce::{ReduceAccumulator, ReduceWorkItem, accumulate_read};
use cubecl::tile::kind::Masked;
use cubecl::{self as cubecl, cube, prelude::*};

/// Line `line` of `lines` accumulated into `acc` as line `index` of its row, `row_lanes` of its lanes
/// holding one row's values.
#[cube]
pub(crate) fn accumulate_line<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    acc: &mut C,
    lines: &Masked<'_, Vector<In, N>, usize>,
    line: usize,
    index: usize,
    #[comptime] row_lanes: usize,
    #[comptime] item: ReduceWorkItem,
) {
    accumulate_read::<In, Acc, N, C>(acc, lines.read(line), index, row_lanes, item);
}
