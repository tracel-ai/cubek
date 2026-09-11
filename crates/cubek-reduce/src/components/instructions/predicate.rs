use super::{ReduceFamily, ReduceInstruction, plane_extremum, ranks_ahead};
use crate::components::{
    instructions::{
        Accumulator, AccumulatorFormat, Item, ReduceOutputMode, ReduceRequirements, ReduceStep,
        SlotCount, Value, ValueOrder, normalize_to_flag,
    },
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Logical reduction over the axis: returns `1` if every element is non-zero
/// (AND), or if any element is (OR), and `0` otherwise.
///
/// Each element is normalized to a `0/1` flag. On `{0, 1}`, AND is exactly `min`
/// and OR is exactly `max`, so the two are one reduction in the two
/// [`ValueOrder`]s and the vectorized plane machinery is reused unchanged.
/// Unlike [`Extremum`](super::Extremum) these never see a NaN, since they rank
/// flags rather than the input, so they carry no NaN handling.
#[derive(Debug, CubeType, Clone)]
pub struct Predicate {
    #[cube(comptime)]
    pub order: ValueOrder,
}

impl ReduceFamily for Predicate {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ValueOrder;
}

#[cube]
impl Predicate {
    /// `1` if every element is non-zero: the flags' minimum.
    pub fn all() -> Predicate {
        Predicate {
            order: ValueOrder::Ascending,
        }
    }

    /// `1` if any element is non-zero: the flags' maximum.
    pub fn any() -> Predicate {
        Predicate {
            order: ValueOrder::Descending,
        }
    }

    /// The flag a slot starts from, which is the one this order ranks last.
    fn identity<E: Numeric>(&self) -> E {
        match comptime!(self.order) {
            ValueOrder::Ascending => E::from_int(1),
            ValueOrder::Descending => E::from_int(0),
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Predicate {
    type SharedAccumulator = Shared<[Vector<P::EA, P::SI>]>;
    type Config = ValueOrder;

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Unpacked(SlotCount::Single))
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        Predicate { order: config }
    }

    fn null_input(this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(this.identity::<P::EI>())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        Accumulator::new_Unpacked(
            Value::new_single(Vector::empty().fill(this.identity::<P::EA>())),
            Value::new_None(),
        )
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let accumulator_item = accumulator.elements().item();
        let flag = normalize_to_flag::<P::EI, P::SI>(item.elements);
        let elements = match reduce_step {
            ReduceStep::Plane => {
                let candidate_item =
                    Vector::cast_from(plane_extremum::<P::EI, P::SI>(this.order, flag));
                select_many(
                    ranks_ahead::<P::EA, P::SI>(this.order, accumulator_item, candidate_item),
                    accumulator_item,
                    candidate_item,
                )
            }
            ReduceStep::Identity => {
                let flag = Vector::cast_from(flag);
                select_many(
                    ranks_ahead::<P::EA, P::SI>(this.order, accumulator_item, flag),
                    accumulator_item,
                    flag,
                )
            }
        };

        accumulator
            .elements_mut()
            .assign(&Value::new_single(elements));
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        // The accumulator already holds 0/1 flags, so ranking them is the reduction.
        let acc_item = accumulator.elements().item();
        let candidate_item =
            Vector::cast_from(plane_extremum::<P::EA, P::SI>(this.order, acc_item));
        let winning = select_many(
            ranks_ahead::<P::EA, P::SI>(this.order, acc_item, candidate_item),
            acc_item,
            candidate_item,
        );
        accumulator
            .elements_mut()
            .assign(&Value::new_single(winning));
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let accumulator_item = accumulator.elements().item();
        let other_item = other.elements().item();

        accumulator
            .elements_mut()
            .assign(&Value::new_single(select_many(
                ranks_ahead::<P::EA, P::SI>(this.order, accumulator_item, other_item),
                accumulator_item,
                other_item,
            )));
    }

    fn output_mode(_this: &Self) -> comptime_type!(ReduceOutputMode) {
        ReduceOutputMode::Values
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        let mut winning = this.identity::<P::EA>();
        let accumulator = accumulator.elements().item();
        #[unroll]
        for k in 0..accumulator.vector_size() {
            let candidate = accumulator.extract(k);
            let ahead = match comptime!(this.order) {
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

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        (
            Value::new_single(Vector::cast_from(accumulator.elements().item())),
            Value::new_None(),
        )
    }
}
