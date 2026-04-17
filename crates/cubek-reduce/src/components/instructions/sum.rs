use super::{ReduceFamily, ReduceInstruction, ReduceRequirements};
use crate::components::{
    instructions::{Accumulator, AccumulatorKind, Item, ReduceStep},
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

    fn from_config(_config: Self::Config) -> Self {
        Sum {}
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::from_int(0))
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(Vector::empty().fill(P::EA::from_int(0))),
            args: AccumulatorKind::new_None(),
        }
    }

    fn assign_accumulator(_this: &Self, destination: &mut Accumulator<P>, source: &Accumulator<P>) {
        destination.elements.assign(&source.elements);
    }

    fn reduce(
        _this: &Self,
        accumulator: &Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) -> Accumulator<P> {
        let accumulator = &accumulator.elements.item();
        let item = item.elements;
        let elements = match reduce_step {
            ReduceStep::Plane => *accumulator + plane_sum(Vector::cast_from(item)),
            ReduceStep::Identity => *accumulator + Vector::cast_from(item),
        };
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(elements),
            args: AccumulatorKind::new_None(),
        }
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let sum = plane_sum(Vector::cast_from(accumulator.elements.item()));
        accumulator
            .elements
            .assign(&AccumulatorKind::new_single(sum));
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: &Accumulator<P>,
        rhs: &Accumulator<P>,
    ) -> Accumulator<P> {
        let lhs = lhs.elements.item();
        let rhs = rhs.elements.item();

        Accumulator::<P> {
            elements: AccumulatorKind::new_single(lhs + rhs),
            args: AccumulatorKind::new_None(),
        }
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        let sum = Vector::vector_sum(accumulator.elements.item());

        AccumulatorKind::new_single(Out::cast_from(sum))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        AccumulatorKind::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
