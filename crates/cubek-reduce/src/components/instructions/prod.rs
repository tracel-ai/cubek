use super::{ReduceFamily, ReduceInstruction};
use crate::components::{
    instructions::{Accumulator, AccumulatorKind, Item, ReduceRequirements, ReduceStep},
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

    fn from_config(_config: Self::Config) -> Self {
        Prod {}
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::from_int(1))
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(Vector::empty().fill(P::EA::from_int(1))),
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
        let item = Vector::cast_from(item.elements);
        let accumulator = &accumulator.elements.item();
        let elements = match reduce_step {
            ReduceStep::Plane => *accumulator * plane_prod(item),
            ReduceStep::Identity => *accumulator * item,
        };

        Accumulator::<P> {
            elements: AccumulatorKind::new_single(elements),
            args: AccumulatorKind::new_None(),
        }
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let prod = plane_prod(Vector::cast_from(accumulator.elements.item()));
        accumulator
            .elements
            .assign(&AccumulatorKind::new_single(prod));
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: &Accumulator<P>,
        rhs: &Accumulator<P>,
    ) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(lhs.elements.item() * rhs.elements.item()),
            args: AccumulatorKind::new_None(),
        }
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        let accumulator = accumulator.elements.item();
        let mut prod = P::EA::from_int(1);
        #[unroll]
        for k in 0..accumulator.size() {
            prod *= accumulator[k];
        }
        AccumulatorKind::new_single(Out::cast_from(prod))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        AccumulatorKind::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
