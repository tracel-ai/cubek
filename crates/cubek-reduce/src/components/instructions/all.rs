use super::{ReduceFamily, ReduceInstruction};
use crate::components::{
    instructions::{
        Accumulator, AccumulatorFormat, Item, ReduceRequirements, ReduceStep, Value,
        normalize_to_flag,
    },
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Logical-AND reduction: returns `1` if every element along the reduced axis is
/// non-zero, and `0` otherwise.
///
/// Each element is normalized to a `0/1` flag, then combined with `min`. On
/// `{0, 1}`, AND is exactly `min`, so the vectorized `plane_min` machinery is
/// reused unchanged.
#[derive(Debug, CubeType, Clone)]
pub struct All;

impl ReduceFamily for All {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for All {
    type SharedAccumulator = Shared<[Vector<P::EA, P::SI>]>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        All {}
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        // 1 normalizes to flag 1, the identity for AND.
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
        let accumulator_item = accumulator.elements.item();
        let flag = normalize_to_flag::<P::EI, P::SI>(item.elements);
        let elements = match reduce_step {
            ReduceStep::Plane => {
                let candidate_item = Vector::cast_from(plane_min(flag));
                select_many(
                    accumulator_item.less_than(&candidate_item),
                    accumulator_item,
                    candidate_item,
                )
            }
            ReduceStep::Identity => {
                let flag = Vector::cast_from(flag);
                select_many(accumulator_item.less_than(&flag), accumulator_item, flag)
            }
        };

        accumulator.elements.assign(&Value::new_single(elements));
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        // The accumulator already holds 0/1 flags, so min-of-flags is AND.
        let acc_item = accumulator.elements.item();
        let candidate_item = Vector::cast_from(plane_min(acc_item));
        let all = select_many(
            acc_item.less_than(&candidate_item),
            acc_item,
            candidate_item,
        );
        accumulator.elements.assign(&Value::new_single(all));
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let accumulator_item = accumulator.elements.item();
        let other_item = other.elements.item();

        accumulator.elements.assign(&Value::new_single(select_many(
            accumulator_item.less_than(&other_item),
            accumulator_item,
            other_item,
        )));
    }

    fn to_output_parallel<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        // Fold the vectorized flags from the AND identity (1).
        let mut all = P::EA::from_int(1);
        let accumulator = accumulator.elements.item();
        #[unroll]
        for k in 0..accumulator.size() {
            let candidate = accumulator.extract(k);
            all = select(candidate < all, candidate, all);
        }
        Value::new_single(Out::cast_from(all))
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        Value::new_single(Vector::cast_from(accumulator.elements.item()))
    }
}
