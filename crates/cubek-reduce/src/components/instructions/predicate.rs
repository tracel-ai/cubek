use super::{plane_extremum, ranks_ahead};
use crate::components::{
    instructions::{
        Accumulator, AccumulatorFormat, Item, ReduceOutputMode, ReduceRequirements, ReduceStep,
        Value, ValueOrder, normalize_to_flag,
    },
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// The implementation [`All`](super::All) and [`Any`](super::Any) share: on
/// `{0, 1}` flags AND is exactly `min` and OR is exactly `max`, so the two are
/// one reduction in the two [`ValueOrder`]s.
///
/// Unlike [`Extremum`](super::Extremum) these never see a NaN, since they rank
/// flags rather than the input, so they carry no NaN handling.
#[derive(Debug, CubeType, Clone)]
pub(crate) struct Predicate {
    #[cube(comptime)]
    pub(crate) order: ValueOrder,
}

#[cube]
impl Predicate {
    /// The flag a slot starts from, which is the one this order ranks last.
    fn identity<E: Numeric>(&self) -> E {
        match comptime!(self.order) {
            ValueOrder::Ascending => E::from_int(1),
            ValueOrder::Descending => E::from_int(0),
        }
    }

    pub(crate) fn requirements(&self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    pub(crate) fn accumulator_format(&self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    pub(crate) fn null_input<P: ReducePrecision>(&self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(self.identity::<P::EI>())
    }

    pub(crate) fn null_accumulator<P: ReducePrecision>(&self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(self.identity::<P::EA>())),
            args: Value::new_None(),
        }
    }

    pub(crate) fn reduce<P: ReducePrecision>(
        &self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let accumulator_item = accumulator.elements.item();
        let flag = normalize_to_flag::<P::EI, P::SI>(item.elements);
        let elements = match reduce_step {
            ReduceStep::Plane => {
                let candidate_item =
                    Vector::cast_from(plane_extremum::<P::EI, P::SI>(self.order, flag));
                select_many(
                    ranks_ahead::<P::EA, P::SI>(self.order, accumulator_item, candidate_item),
                    accumulator_item,
                    candidate_item,
                )
            }
            ReduceStep::Identity => {
                let flag = Vector::cast_from(flag);
                select_many(
                    ranks_ahead::<P::EA, P::SI>(self.order, accumulator_item, flag),
                    accumulator_item,
                    flag,
                )
            }
        };

        accumulator.elements.assign(&Value::new_single(elements));
    }

    pub(crate) fn plane_reduce_inplace<P: ReducePrecision>(
        &self,
        accumulator: &mut Accumulator<P>,
    ) {
        // The accumulator already holds 0/1 flags, so ranking them is the reduction.
        let acc_item = accumulator.elements.item();
        let candidate_item =
            Vector::cast_from(plane_extremum::<P::EA, P::SI>(self.order, acc_item));
        let winning = select_many(
            ranks_ahead::<P::EA, P::SI>(self.order, acc_item, candidate_item),
            acc_item,
            candidate_item,
        );
        accumulator.elements.assign(&Value::new_single(winning));
    }

    pub(crate) fn fuse_accumulators<P: ReducePrecision>(
        &self,
        accumulator: &mut Accumulator<P>,
        other: &Accumulator<P>,
    ) {
        let accumulator_item = accumulator.elements.item();
        let other_item = other.elements.item();

        accumulator.elements.assign(&Value::new_single(select_many(
            ranks_ahead::<P::EA, P::SI>(self.order, accumulator_item, other_item),
            accumulator_item,
            other_item,
        )));
    }

    pub(crate) fn output_mode(&self) -> comptime_type!(ReduceOutputMode) {
        ReduceOutputMode::Values
    }

    pub(crate) fn to_output_parallel<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        accumulator: Accumulator<P>,
    ) -> (Value<Out>, Value<Idx>) {
        let mut winning = self.identity::<P::EA>();
        let accumulator = accumulator.elements.item();
        #[unroll]
        for k in 0..accumulator.vector_size() {
            let candidate = accumulator.extract(k);
            let ahead = match comptime!(self.order) {
                ValueOrder::Ascending => candidate < winning,
                ValueOrder::Descending => candidate > winning,
            };
            winning = select(ahead, candidate, winning);
        }
        (
            Value::new_single(Out::cast_from(winning)),
            Value::new_None(),
        )
    }

    pub(crate) fn to_output_perpendicular<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        accumulator: Accumulator<P>,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        (
            Value::new_single(Vector::cast_from(accumulator.elements.item())),
            Value::new_None(),
        )
    }
}
