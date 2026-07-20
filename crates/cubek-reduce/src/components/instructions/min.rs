use super::{ArgAccumulator, ReduceFamily, ReduceInstruction, lowest_coordinate_matching};
use crate::components::{
    instructions::{
        Accumulator, AccumulatorFormat, Item, ReduceOutputMode, ReduceRequirements, ReduceStep,
        ReduceWithIndices, ReduceWithIndicesFamily, Value,
    },
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Return the minimum item, its coordinate, or both, per [`ReduceOutputMode`].
/// In case of equality, the lowest coordinate is selected.
#[derive(Debug, CubeType, Clone)]
pub struct Min {
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for Min {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

impl ReduceWithIndicesFamily for Min {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

#[cube]
impl Min {
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
            items0.equal(&items1),
            coordinates0.less_than(&coordinates1),
            items0.less_than(&items1),
        );
        let items = select_many(to_keep, items0, items1);
        let coordinates = select_many(to_keep, coordinates0, coordinates1);
        (items, coordinates)
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Min {
    type SharedAccumulator = ArgAccumulator<P>;
    type Config = ReduceOutputMode;

    fn requirements(this: &Self) -> ReduceRequirements {
        ReduceRequirements {
            coordinates: comptime!(this.output.has_indices()),
        }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        Min { output: config }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::max_value())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let args = if comptime!(this.output.has_indices()) {
            Value::new_single(Vector::empty().fill(u32::MAX))
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(P::EA::max_value())),
            args,
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let has_indices = comptime!(this.output.has_indices());

        if has_indices {
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
        } else {
            let accumulator_item = accumulator.elements.item();
            let item = item.elements;
            let elements = match reduce_step {
                ReduceStep::Plane => {
                    let candidate_item = Vector::cast_from(plane_min(item));
                    select_many(
                        accumulator_item.less_than(&candidate_item),
                        accumulator_item,
                        candidate_item,
                    )
                }
                ReduceStep::Identity => {
                    let item = Vector::cast_from(item);
                    select_many(accumulator_item.less_than(&item), accumulator_item, item)
                }
            };

            accumulator.elements.assign(&Value::new_single(elements));
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        let has_indices = comptime!(this.output.has_indices());

        if has_indices {
            let acc_item = accumulator.elements.item();
            let coordinate = accumulator.args.item();

            let candidate_item = plane_min(acc_item);
            let candidate_coordinate =
                lowest_coordinate_matching(candidate_item, acc_item, coordinate);

            let (elements, args) = Self::choose_argmin(
                accumulator.elements.item(),
                accumulator.args.item(),
                Vector::cast_from(candidate_item),
                candidate_coordinate,
            );

            accumulator.elements.assign(&Value::new_single(elements));
            accumulator.args.assign(&Value::new_single(args));
        } else {
            let acc_item = accumulator.elements.item();
            let candidate_item = Vector::cast_from(plane_min(acc_item));
            let min = select_many(
                acc_item.less_than(&candidate_item),
                acc_item,
                candidate_item,
            );
            accumulator.elements.assign(&Value::new_single(min));
        }
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let has_indices = comptime!(this.output.has_indices());

        if has_indices {
            let (elements, args) = Self::choose_argmin(
                accumulator.elements.item(),
                accumulator.args.item(),
                other.elements.item(),
                other.args.item(),
            );

            accumulator.elements.assign(&Value::new_single(elements));
            accumulator.args.assign(&Value::new_single(args));
        } else {
            let accumulator_item = accumulator.elements.item();
            let other_item = other.elements.item();

            accumulator.elements.assign(&Value::new_single(select_many(
                accumulator_item.less_than(&other_item),
                accumulator_item,
                other_item,
            )));
        }
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let has_indices = comptime!(this.output.has_indices());

        if has_indices {
            let (_min, coordinate) = min_finalize_with_coords::<P>(&accumulator);
            Value::new_single(Out::cast_from(coordinate))
        } else {
            let mut min = P::EA::max_value();
            let accumulator = accumulator.elements.item();
            #[unroll]
            for k in 0..accumulator.size() {
                let candidate = accumulator.extract(k);
                min = select(candidate < min, candidate, min);
            }
            Value::new_single(Out::cast_from(min))
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        let has_indices = comptime!(this.output.has_indices());

        if has_indices {
            Value::new_single(Vector::cast_from(accumulator.args.item()))
        } else {
            Value::new_single(Vector::cast_from(accumulator.elements.item()))
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceWithIndices<P> for Min {
    fn to_output_both_parallel<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        let (min, coordinate) = min_finalize_with_coords::<P>(&accumulator);
        (
            Value::new_single(Out::cast_from(min)),
            Value::new_single(Idx::cast_from(coordinate)),
        )
    }

    fn to_output_both_perpendicular<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        (
            Value::new_single(Vector::cast_from(accumulator.elements.item())),
            Value::new_single(Vector::cast_from(accumulator.args.item())),
        )
    }
}

/// Collapse the vectorized accumulator lanes down to the final minimum and its
/// coordinate, for the parallel layout.
///
/// Ties break towards the lower coordinate, matching the CPU reference. The
/// accumulator must have been built with coordinate tracking on.
#[cube]
fn min_finalize_with_coords<P: ReducePrecision>(accumulator: &Accumulator<P>) -> (P::EA, u32) {
    let vector_size = accumulator.elements.item().size().comptime();

    if vector_size > 1 {
        let mut min = P::EA::max_value();
        let mut coordinate = u32::MAX.runtime();

        #[unroll]
        for k in 0..vector_size {
            let acc_element = accumulator.elements.item().extract(k);
            let acc_coordinate = accumulator.args.item().extract(k);

            let take = select(
                acc_element == min,
                acc_coordinate < coordinate,
                acc_element < min,
            );

            min = select(take, acc_element, min);
            coordinate = select(take, acc_coordinate, coordinate);
        }

        (min, coordinate)
    } else {
        (
            accumulator.elements.item().extract(0),
            accumulator.args.item().extract(0),
        )
    }
}
