use super::{ReduceFamily, ReduceInstruction};
use crate::components::{
    instructions::{Accumulator, AccumulatorKind, Item, ReduceRequirements, ReduceStep},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

// TODO Add to test framework.
/// Return the item with the maximum absolute value.
#[derive(Debug, CubeType, Clone)]
pub struct Min;

impl ReduceFamily for Min {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Min {
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        Min {}
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::max_value())
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(Vector::empty().fill(P::EA::max_value())),
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
            ReduceStep::Plane => {
                let candidate_item = Vector::cast_from(plane_min(item));
                select_many(
                    accumulator.less_than(candidate_item),
                    *accumulator,
                    candidate_item,
                )
            }
            ReduceStep::Identity => {
                let item = Vector::cast_from(item);
                select_many(accumulator.less_than(item), *accumulator, item)
            }
        };

        Accumulator::<P> {
            elements: AccumulatorKind::new_single(elements),
            args: AccumulatorKind::new_None(),
        }
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let acc_item = accumulator.elements.item();
        let candidate_item = Vector::cast_from(plane_min(acc_item));
        let min = select_many(acc_item.less_than(candidate_item), acc_item, candidate_item);
        accumulator
            .elements
            .assign(&AccumulatorKind::new_single(min));
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: &Accumulator<P>,
        rhs: &Accumulator<P>,
    ) -> Accumulator<P> {
        let lhs = lhs.elements.item();
        let rhs = rhs.elements.item();
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(select_many(lhs.less_than(rhs), lhs, rhs)),
            args: AccumulatorKind::new_None(),
        }
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        let mut min = P::EA::max_value();
        let accumulator = accumulator.elements.item();
        #[unroll]
        for k in 0..accumulator.size() {
            let candidate = accumulator[k];
            min = select(candidate < min, candidate, min);
        }
        AccumulatorKind::new_single(Out::cast_from(min))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        AccumulatorKind::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
