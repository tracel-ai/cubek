use super::{
    ArgAccumulator, ReduceFamily, ReduceInstruction, ReduceRequirements, lowest_coordinate_matching,
};
use crate::components::{
    instructions::{Accumulator, AccumulatorFormat, Item, ReduceStep, Value},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Compute the coordinate of the maximum item returning the smallest coordinate in case of equality.
#[derive(Debug, CubeType, Clone)]
pub struct ArgMin {}

impl ReduceFamily for ArgMin {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl ArgMin {
    /// Compare two pairs of items and coordinates and return a new pair
    /// where each element in the vectors is the minimal item with its coordinate.
    /// In case of equality, the lowest coordinate is selected.
    pub fn choose_argmin<T: Numeric, N: Size>(
        items0: Vector<T, N>,
        coordinates0: Vector<u32, N>,
        items1: Vector<T, N>,
        coordinates1: Vector<u32, N>,
    ) -> (Vector<T, N>, Vector<u32, N>) {
        let to_keep = select_many(
            items0.equal(items1),
            coordinates0.less_than(coordinates1),
            items0.less_than(items1),
        );
        let items = select_many(to_keep, items0, items1);
        let coordinates = select_many(to_keep, coordinates0, coordinates1);
        (items, coordinates)
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ArgMin {
    type SharedAccumulator = ArgAccumulator<P>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: true }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        ArgMin {}
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::max_value())
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(P::EA::max_value())),
            args: Value::new_single(Vector::empty().fill(u32::MAX)),
        }
    }

    fn reduce(
        _this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let coordinate = item.args.item();
        let item = item.elements;

        let (candidate_item, candidate_coordinate) = match reduce_step {
            ReduceStep::Plane => {
                let candidate_item = plane_min(item);
                let candidate_coordinate =
                    lowest_coordinate_matching(candidate_item, item, coordinate);
                (candidate_item, candidate_coordinate)
            }
            ReduceStep::Identity => (item, coordinate),
        };

        let (elements, args) = Self::choose_argmin(
            accumulator.elements.item(),
            accumulator.args.item(),
            Vector::cast_from(candidate_item),
            candidate_coordinate,
        );

        accumulator.elements.assign(&Value::new_single(elements));
        accumulator.args.assign(&Value::new_single(args));
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let acc_item = accumulator.elements.item();
        let coordinate = accumulator.args.item();

        let candidate_item = plane_min(acc_item);
        let candidate_coordinate = lowest_coordinate_matching(candidate_item, acc_item, coordinate);

        let (elements, args) = Self::choose_argmin(
            accumulator.elements.item(),
            accumulator.args.item(),
            Vector::cast_from(candidate_item),
            candidate_coordinate,
        );

        accumulator.elements.assign(&Value::new_single(elements));
        accumulator.args.assign(&Value::new_single(args));
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let (elements, args) = Self::choose_argmin(
            accumulator.elements.item(),
            accumulator.args.item(),
            other.elements.item(),
            other.args.item(),
        );

        accumulator.elements.assign(&Value::new_single(elements));
        accumulator.args.assign(&Value::new_single(args));
    }

    fn to_output_parallel<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let vector_size = accumulator.elements.item().size().comptime();
        let value = if vector_size > 1 {
            let mut min = P::EA::max_value();
            let mut coordinate = u32::MAX.runtime();

            #[unroll]
            for k in 0..vector_size {
                let acc_element = accumulator.elements.item()[k];
                let acc_coordinate = accumulator.args.item()[k];
                // TODO replace with select
                if acc_element == min && acc_coordinate < coordinate {
                    coordinate = acc_coordinate;
                } else if acc_element < min {
                    min = acc_element;
                    coordinate = acc_coordinate;
                }
            }
            Out::cast_from(coordinate)
        } else {
            Out::cast_from(accumulator.args.item())
        };

        Value::new_single(value)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        Value::new_single(Vector::cast_from(accumulator.args.item()))
    }
}
