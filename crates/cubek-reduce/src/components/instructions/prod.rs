use super::{ReduceFamily, ReduceInstruction};
use crate::components::{
    instructions::{Accumulator, AccumulatorFormat, Item, ReduceRequirements, ReduceStep, Value},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

#[derive(Debug, CubeType, Clone)]
pub struct Prod {}

impl ReduceFamily for Prod {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Prod {
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        Prod {}
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::from_int(1))
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(P::EA::from_int(1))),
            args: Value::new_None(),
        }
    }

    fn reduce(
        _this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let item = Vector::cast_from(item.elements);
        let accumulator_item = &accumulator.elements.item();
        let elements = match reduce_step {
            ReduceStep::Plane => *accumulator_item * plane_prod(item),
            ReduceStep::Identity => *accumulator_item * item,
        };

        accumulator.elements.assign(&Value::new_single(elements));
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let prod = plane_prod(Vector::cast_from(accumulator.elements.item()));
        accumulator.elements.assign(&Value::new_single(prod));
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let accumulator_item = accumulator.elements.item();
        let other_item = other.elements.item();

        accumulator
            .elements
            .assign(&Value::new_single(accumulator_item * other_item));
    }

    fn to_output_parallel<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let accumulator = accumulator.elements.item();
        let mut prod = P::EA::from_int(1);
        #[unroll]
        for k in 0..accumulator.size() {
            prod *= accumulator[k];
        }
        Value::new_single(Out::cast_from(prod))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        Value::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
