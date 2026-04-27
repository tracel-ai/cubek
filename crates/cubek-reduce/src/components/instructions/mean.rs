use super::{ReduceFamily, ReduceInstruction, ReduceRequirements, Sum};
use crate::components::{
    instructions::{Accumulator, AccumulatorFormat, Item, ReduceStep, Value},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

#[derive(Debug, CubeType, Clone)]
pub struct Mean {
    pub(crate) sum: Sum,
}

impl ReduceFamily for Mean {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
fn null_input<P: ReducePrecision, SI: ReduceInstruction<P>>(sum: &SI) -> Vector<P::EI, P::SI> {
    SI::null_input(sum)
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Mean {
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        Mean { sum: Sum {} }
    }

    fn null_input(this: &Self) -> Vector<P::EI, P::SI> {
        <Sum as ReduceInstruction<P>>::null_input(&this.sum)
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        <Sum as ReduceInstruction<P>>::null_accumulator(&this.sum)
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        <Sum as ReduceInstruction<P>>::reduce(&this.sum, accumulator, item, reduce_step)
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        <Sum as ReduceInstruction<P>>::plane_reduce_inplace(&this.sum, accumulator)
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        <Sum as ReduceInstruction<P>>::fuse_accumulators(&this.sum, accumulator, other)
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: VectorSize,
    ) -> Value<Out> {
        let sum = <Sum as ReduceInstruction<P>>::to_output_parallel::<P::EA>(
            &this.sum,
            accumulator,
            shape_axis_reduce,
        )
        .item();

        let value = Out::cast_from(sum / P::EA::cast_from(shape_axis_reduce));
        Value::new_single(value)
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: VectorSize,
    ) -> Value<Vector<Out, P::SI>> {
        let sum = <Sum as ReduceInstruction<P>>::to_output_perpendicular::<P::EA>(
            &this.sum,
            accumulator,
            shape_axis_reduce,
        )
        .item();

        let vector = Vector::cast_from(sum / Vector::cast_from(shape_axis_reduce));
        Value::new_single(vector)
    }
}
