use super::{ReduceFamily, ReduceInstruction, ReduceRequirements};
use crate::components::{
    instructions::{Accumulator, AccumulatorFormat, Item, ReduceStep, Value},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

#[derive(Debug, CubeType, Clone)]
pub struct Sum {}

impl ReduceFamily for Sum {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Sum {
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        Sum {}
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
        let accumulator_item = &accumulator.elements.item();
        let item = item.elements;
        let elements = match reduce_step {
            ReduceStep::Plane => *accumulator_item + plane_sum(Vector::cast_from(item)),
            ReduceStep::Identity => *accumulator_item + Vector::cast_from(item),
        };

        accumulator.elements.assign(&Value::new_single(elements));
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let sum = plane_sum(Vector::cast_from(accumulator.elements.item()));
        accumulator.elements.assign(&Value::new_single(sum));
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let accumulator_item = accumulator.elements.item();
        let other_item = other.elements.item();

        accumulator
            .elements
            .assign(&Value::new_single(accumulator_item + other_item));
    }

    fn to_output_parallel<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let sum = Vector::vector_sum(accumulator.elements.item());

        Value::new_single(Out::cast_from(sum))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        Value::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
