use super::{ReduceFamily, ReduceInstruction};
use crate::components::{
    instructions::{Accumulator, AccumulatorFormat, Item, ReduceRequirements, ReduceStep, Value},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

// TODO Add to test framework.
/// Return the item with the maximum absolute value.
#[derive(Debug, CubeType, Clone)]
pub struct MaxAbs;

impl ReduceFamily for MaxAbs {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for MaxAbs {
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        MaxAbs {}
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::from_int(0))
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(P::EA::from_int(0))),
            args: Value::new_None(),
        }
    }

    fn reduce(
        _this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let accumulator_item = accumulator.elements.item();
        let elements = match reduce_step {
            ReduceStep::Plane => {
                let candidate_item = Vector::cast_from(plane_max(Vector::abs(item.elements)));
                select_many(
                    accumulator_item.greater_than(candidate_item),
                    accumulator_item,
                    candidate_item,
                )
            }
            ReduceStep::Identity => {
                let item_abs = Vector::cast_from(Vector::abs(item.elements));
                select_many(
                    accumulator_item.greater_than(item_abs),
                    accumulator_item,
                    item_abs,
                )
            }
        };

        accumulator.elements.assign(&Value::new_single(elements));
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let accumulator_item = accumulator.elements.item();
        let other_item = other.elements.item();

        accumulator.elements.assign(&Value::new_single(select_many(
            accumulator_item.greater_than(other_item),
            accumulator_item,
            other_item,
        )));
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let acc_item = accumulator.elements.item();
        let candidate_item = Vector::cast_from(plane_max(Vector::abs(acc_item)));
        let max = select_many(
            acc_item.greater_than(candidate_item),
            acc_item,
            candidate_item,
        );
        accumulator.elements.assign(&Value::new_single(max));
    }

    fn to_output_parallel<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let mut max = P::EA::from_int(0);
        let accumulator = accumulator.elements.item();
        #[unroll]
        for k in 0..accumulator.size() {
            let candidate = accumulator[k];
            max = select(candidate > max, candidate, max);
        }
        Value::new_single(Out::cast_from(max))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        Value::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
