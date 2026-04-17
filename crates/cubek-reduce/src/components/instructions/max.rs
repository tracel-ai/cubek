use super::{ReduceFamily, ReduceInstruction};
use crate::components::{
    instructions::{Accumulator, AccumulatorKind, Item, ReduceRequirements, ReduceStep},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

// TODO Add to test framework.
/// Return the item with the maximum value.
#[derive(Debug, CubeType, Clone)]
pub struct Max;

impl ReduceFamily for Max {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Max {
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        Max {}
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::min_value())
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(Vector::empty().fill(P::EA::min_value())),
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
        let accumulator = accumulator.elements.item();
        let elements = match reduce_step {
            ReduceStep::Plane => {
                let candidate_item = Vector::cast_from(plane_max(item.elements));
                select_many(
                    accumulator.greater_than(candidate_item),
                    accumulator,
                    candidate_item,
                )
            }
            ReduceStep::Identity => {
                let item = Vector::cast_from(item.elements);
                select_many(accumulator.greater_than(item), accumulator, item)
            }
        };

        Accumulator::<P> {
            elements: AccumulatorKind::new_single(elements),
            args: AccumulatorKind::new_None(),
        }
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let acc_item = accumulator.elements.item();
        let candidate_item = Vector::cast_from(plane_max(acc_item));
        let max = select_many(
            acc_item.greater_than(candidate_item),
            acc_item,
            candidate_item,
        );
        accumulator
            .elements
            .assign(&AccumulatorKind::new_single(max));
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: &Accumulator<P>,
        rhs: &Accumulator<P>,
    ) -> Accumulator<P> {
        let lhs = lhs.elements.item();
        let rhs = rhs.elements.item();
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(select_many(lhs.greater_than(rhs), lhs, rhs)),
            args: AccumulatorKind::new_None(),
        }
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        let mut max = P::EA::min_value();
        let accumulator = accumulator.elements.item();
        #[unroll]
        for k in 0..accumulator.size() {
            let candidate = accumulator[k];
            max = select(candidate > max, candidate, max);
        }
        AccumulatorKind::new_single(Out::cast_from(max))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        AccumulatorKind::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
