use super::{ReduceCoordinate, ReduceFamily, ReduceInstruction};
use crate::{components::instructions::ReduceRequirements, components::precision::ReducePrecision};
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
    type AccumulatorItem = Vector<P::EA, P::SI>;
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

    fn null_accumulator(_this: &Self) -> Self::AccumulatorItem {
        Vector::empty().fill(P::EA::max_value())
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        *destination = *source;
    }

    fn read_accumulator(
        _this: &Self,
        accumulator: &Vector<P::EA, P::SI>,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        (
            Vector::cast_from(*accumulator),
            ReduceCoordinate::new_NotRequired(),
        )
    }

    fn reduce(
        _this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Vector<P::EI, P::SI>,
        _coordinate: ReduceCoordinate<P::SI>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        if use_planes {
            let candidate_item = Vector::cast_from(plane_min(item));
            select_many(
                accumulator.less_than(candidate_item),
                *accumulator,
                candidate_item,
            )
        } else {
            let item = Vector::cast_from(item);
            select_many(accumulator.less_than(item), *accumulator, item)
        }
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        select_many(lhs.less_than(rhs), lhs, rhs)
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Out {
        let mut min = P::EA::max_value();
        #[unroll]
        for k in 0..accumulator.size() {
            let candidate = accumulator[k];
            min = select(candidate < min, candidate, min);
        }
        Out::cast_from(min)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Vector<Out, P::SI> {
        Vector::cast_from(accumulator)
    }
}
