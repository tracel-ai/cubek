use super::{
    ArgAccumulator, ReduceFamily, ReduceInstruction, max_identity, plane_argmax_propagating_nan,
    plane_max_propagating_nan, select_argmax, select_max,
};
use crate::components::{
    instructions::{
        Accumulator, AccumulatorFormat, Item, ReduceOutputMode, ReduceRequirements, ReduceStep,
        ReduceWithIndices, ReduceWithIndicesFamily, Value, ValueExpand,
    },
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Return the maximum item, its coordinate, or both, per [`ReduceOutputMode`].
/// NaNs take precedence over non-NaN values. When indices are returned, ties
/// and multiple NaNs select the lowest coordinate.
#[derive(Debug, CubeType, Clone)]
pub struct Max {
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for Max {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

impl ReduceWithIndicesFamily for Max {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

/// Fold `candidate` into the accumulator, keeping the larger item per vector
/// element (and its coordinate, when the candidate carries one).
///
/// Ties break towards the lower coordinate, matching the CPU reference. A
/// coordinate-less candidate emits no index arithmetic at all.
#[cube]
fn max_insert<T: Numeric, N: Size>(
    elements: &mut Value<Vector<T, N>>,
    coordinates: &mut Value<Vector<u32, N>>,
    candidate: Vector<T, N>,
    candidate_coord: &Value<Vector<u32, N>>,
) {
    let acc = elements.item();

    match candidate_coord {
        Value::None => elements.assign(&Value::new_single(select_max(acc, candidate))),
        Value::Single(coord) => {
            let candidate_coord = coord.unwrap();
            let acc_coord = coordinates.item();
            let (selected, selected_coord) =
                select_argmax(acc, acc_coord, candidate, candidate_coord);
            elements.assign(&Value::new_single(selected));
            coordinates.assign(&Value::new_single(selected_coord));
        }
        Value::Multiple(_) => panic!("a max candidate carries at most one coordinate"),
    }
}

/// Reduce `item` across the plane to one winning candidate, with its lowest
/// matching coordinate when the item carries one.
#[cube]
fn plane_max_candidate<T: Numeric, N: Size>(
    item: Vector<T, N>,
    coordinates: &Value<Vector<u32, N>>,
) -> (Vector<T, N>, Value<Vector<u32, N>>) {
    match coordinates {
        Value::None => (plane_max_propagating_nan(item), Value::new_None()),
        Value::Single(coord) => {
            let (winning, winning_coord) = plane_argmax_propagating_nan(item, coord.unwrap());
            (winning, Value::new_single(winning_coord))
        }
        Value::Multiple(_) => panic!("a max candidate carries at most one coordinate"),
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Max {
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
        Max { output: config }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(max_identity::<P::EI>())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let args = if comptime!(this.output.has_indices()) {
            Value::new_single(Vector::empty().fill(u32::MAX))
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(max_identity::<P::EA>())),
            args,
        }
    }

    fn reduce(
        _this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let (candidate, candidate_coord) = match reduce_step {
            ReduceStep::Plane => plane_max_candidate(item.elements, &item.args),
            ReduceStep::Identity => (item.elements, item.args),
        };

        max_insert(
            &mut accumulator.elements,
            &mut accumulator.args,
            Vector::cast_from(candidate),
            &candidate_coord,
        );
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let (candidate, candidate_coord) =
            plane_max_candidate(accumulator.elements.item(), &accumulator.args);

        max_insert(
            &mut accumulator.elements,
            &mut accumulator.args,
            candidate,
            &candidate_coord,
        );
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        max_insert(
            &mut accumulator.elements,
            &mut accumulator.args,
            other.elements.item(),
            &other.args,
        );
    }

    fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode) {
        comptime!(this.output)
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        match accumulator.args {
            Value::None => {
                let acc = accumulator.elements.item();
                let mut max = max_identity::<P::EA>();
                #[unroll]
                for k in 0..acc.size() {
                    let candidate = acc.extract(k);
                    max = select_max(
                        Vector::<P::EA, Const<1>>::new(candidate),
                        Vector::<P::EA, Const<1>>::new(max),
                    )
                    .extract(0);
                }
                (Value::new_single(Out::cast_from(max)), Value::new_None())
            }
            Value::Single(_) => {
                let (max, coordinate) = max_finalize_with_coords::<P>(&accumulator);
                (
                    Value::new_single(Out::cast_from(max)),
                    Value::new_single(Idx::cast_from(coordinate)),
                )
            }
            Value::Multiple(_) => panic!("a max accumulator holds at most one coordinate vector"),
        }
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        let values = Value::new_single(Vector::cast_from(accumulator.elements.item()));
        let indices = match accumulator.args {
            Value::None => Value::new_None(),
            Value::Single(coord) => Value::new_single(Vector::cast_from(coord.unwrap())),
            Value::Multiple(_) => panic!("a max accumulator holds at most one coordinate vector"),
        };
        (values, indices)
    }
}

impl<P: ReducePrecision> ReduceWithIndices<P> for Max {}

/// Collapse the vectorized accumulator lanes down to the final maximum and its
/// coordinate, for the parallel layout.
///
/// Ties break towards the lower coordinate, matching the CPU reference. The
/// accumulator must have been built with coordinate tracking on.
#[cube]
fn max_finalize_with_coords<P: ReducePrecision>(accumulator: &Accumulator<P>) -> (P::EA, u32) {
    let vector_size = accumulator.elements.item().size().comptime();

    if vector_size > 1 {
        let mut max = max_identity::<P::EA>();
        let mut coordinate = u32::MAX.runtime();

        #[unroll]
        for k in 0..vector_size {
            let acc_element = accumulator.elements.item().extract(k);
            let acc_coordinate = accumulator.args.item().extract(k);

            let (selected, selected_coordinate) = select_argmax(
                Vector::<P::EA, Const<1>>::new(max),
                Vector::<u32, Const<1>>::new(coordinate),
                Vector::<P::EA, Const<1>>::new(acc_element),
                Vector::<u32, Const<1>>::new(acc_coordinate),
            );

            max = selected.extract(0);
            coordinate = selected_coordinate.extract(0);
        }

        (max, coordinate)
    } else {
        (
            accumulator.elements.item().extract(0),
            accumulator.args.item().extract(0),
        )
    }
}
