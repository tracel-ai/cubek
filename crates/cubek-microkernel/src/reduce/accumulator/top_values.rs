//! The `K` best values without their indices: a top-k that writes only its values.

use crate::reduce::{ReduceAccumulator, ReduceCarrierFamily};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// [`ReduceCarrier::TopValues`](crate::reduce::ReduceCarrier::TopValues): its carrier is [`TopValues`].
pub struct TopValuesReduceCarrierFamily;

impl ReduceCarrierFamily for TopValuesReduceCarrierFamily {
    type Carrier<Acc: Numeric, N: Size, K: Size> = TopValues<Acc, N, K>;
}

#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopValues<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, monoid);
    }

    fn value_line(acc: &Self, slot: usize) -> Vector<Acc, N> {
        acc.values[slot]
    }

    fn index_line(_acc: &Self, _slot: usize) -> Vector<u32, N> {
        Vector::new(0u32)
    }

    fn set_line(acc: &mut Self, slot: usize, value: Vector<Acc, N>, _index: Vector<u32, N>) {
        acc.values[slot] = value;
    }

    /// Every lane's list inserted into one one-lane list, best first.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let mut best = TopValues::<Acc, Const<1>, K>::identity(monoid);
        #[unroll]
        for lane in 0..N::value() {
            #[unroll(unrolled)]
            for slot in 0..k {
                best.insert(Vector::new(acc.values[slot].extract(lane)), monoid);
            }
        }

        let mut kept_values = Array::new(k);
        #[unroll]
        for slot in 0..k {
            kept_values[slot] = Vector::new(best.values[slot].extract(0usize));
        }
        TopValues::<Acc, N, K>::new(kept_values)
    }
}
