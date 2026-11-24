//! A value with the index it was read at: the argmax and the argmin.

use crate::reduce::{OneIndexed, ReduceAccumulator, ReduceCarrierFamily, lane_indices};
use cubecl::tile::{Carrier, Indexed, Monoid};
use cubecl::{self as cubecl, cube, prelude::*};

/// [`ReduceCarrier::Indexed`](crate::reduce::ReduceCarrier::Indexed): its carrier is [`Indexed`].
pub struct IndexedReduceCarrierFamily;

impl ReduceCarrierFamily for IndexedReduceCarrierFamily {
    type Carrier<Acc: Numeric, N: Size, K: Size> = Indexed<Acc, N>;
}

#[cube]
impl<Acc: Numeric, N: Size> ReduceAccumulator<Acc, N> for Indexed<Acc, N> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        let read = Indexed::<Acc, N> {
            value: line,
            index: lane_indices::<N>(line_index, row_lanes),
        };
        *acc = Indexed::<Acc, N>::combine_later(*acc, read, monoid);
    }

    fn value_line(acc: &Self, _slot: usize) -> Vector<Acc, N> {
        acc.value
    }

    fn index_line(acc: &Self, _slot: usize) -> Vector<u32, N> {
        acc.index
    }

    fn set_line(acc: &mut Self, _slot: usize, value: Vector<Acc, N>, index: Vector<u32, N>) {
        acc.value = value;
        acc.index = index;
    }

    /// Its lanes reduced one by one as one-lane carriers under `monoid`.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self {
        let mut best = OneIndexed::<Acc>::identity(monoid);
        #[unroll]
        for lane in 0..N::value() {
            let one = OneIndexed::<Acc> {
                value: Vector::new(acc.value.extract(lane)),
                index: Vector::new(acc.index.extract(lane)),
            };
            best = OneIndexed::<Acc>::combine(best, one, monoid);
        }
        Indexed::<Acc, N> {
            value: Vector::new(best.value.extract(0usize)),
            index: Vector::new(best.index.extract(0usize)),
        }
    }
}
