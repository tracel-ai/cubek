use super::{
    ArgAccumulator, ReduceFamily, ReduceInstruction, max_identity, plane_argmax_propagating_nan,
    plane_max_propagating_nan, select_argmax, select_max,
};
use crate::components::{
    instructions::{
        Accumulator, AccumulatorExpand, AccumulatorFormat, Item, PackedExtremum, ReduceOutputMode,
        ReduceRequirements, ReduceStep, ReduceWithIndices, ReduceWithIndicesFamily, SlotCount,
        Value, ValueExpand, packs_key,
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

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        let packs = packs_key::<P>(this.output);

        comptime!(if packs {
            AccumulatorFormat::Packed(SlotCount::Single)
        } else {
            AccumulatorFormat::Unpacked(SlotCount::Single)
        })
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        Max { output: config }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(max_identity::<P::EI>())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let packs = packs_key::<P>(this.output);

        if comptime!(packs) {
            PackedExtremum::descending().null_accumulator::<P>(max_identity::<P::EA>())
        } else {
            let args = if comptime!(this.output.has_indices()) {
                Value::new_single(Vector::empty().fill(u32::MAX))
            } else {
                Value::new_None()
            };

            Accumulator::new_Unpacked(
                Value::new_single(Vector::empty().fill(max_identity::<P::EA>())),
                args,
            )
        }
    }

    fn reduce(
        _this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        match accumulator {
            Accumulator::Packed(keys) => {
                PackedExtremum::descending().reduce::<P>(keys, item, reduce_step)
            }
            Accumulator::Unpacked { elements, args } => {
                let (candidate, candidate_coord) = match reduce_step {
                    ReduceStep::Plane => plane_max_candidate(item.elements, &item.args),
                    ReduceStep::Identity => (item.elements, item.args),
                };

                max_insert(
                    elements,
                    args,
                    Vector::cast_from(candidate),
                    &candidate_coord,
                );
            }
        }
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        match accumulator {
            Accumulator::Packed(keys) => {
                PackedExtremum::descending().plane_reduce_inplace::<P>(keys)
            }
            Accumulator::Unpacked { elements, args } => {
                let (candidate, candidate_coord) = plane_max_candidate(elements.item(), &*args);
                max_insert(elements, args, candidate, &candidate_coord);
            }
        }
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        match (accumulator, other) {
            (Accumulator::Packed(keys), Accumulator::Packed(other_keys)) => {
                PackedExtremum::descending().fuse_accumulators::<P>(keys, other_keys.item())
            }
            (
                Accumulator::Unpacked { elements, args },
                Accumulator::Unpacked {
                    elements: other_elements,
                    args: other_args,
                },
            ) => max_insert(elements, args, other_elements.item(), other_args),
            _ => panic!("both accumulators must hold the same representation"),
        }
    }

    fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode) {
        comptime!(this.output)
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        match accumulator {
            Accumulator::Packed(keys) => {
                PackedExtremum::descending().to_output_parallel::<P, Out, Idx>(keys.item())
            }
            Accumulator::Unpacked { elements, args } => match args {
                Value::None => {
                    let acc = elements.item();
                    let mut max = max_identity::<P::EA>();
                    #[unroll]
                    for k in 0..acc.vector_size() {
                        let candidate = acc.extract(k);
                        max = select_max(
                            Vector::<P::EA, Const<1>>::new(candidate),
                            Vector::<P::EA, Const<1>>::new(max),
                        )
                        .extract(0usize);
                    }
                    (Value::new_single(Out::cast_from(max)), Value::new_None())
                }
                Value::Single(_) => {
                    let (max, coordinate) = max_finalize_with_coords::<P>(&elements, &args);
                    (
                        Value::new_single(Out::cast_from(max)),
                        Value::new_single(Idx::cast_from(coordinate)),
                    )
                }
                Value::Multiple(_) => {
                    panic!("a max accumulator holds at most one coordinate vector")
                }
            },
        }
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        match accumulator {
            Accumulator::Packed(keys) => {
                PackedExtremum::descending().to_output_perpendicular::<P, Out, Idx>(keys.item())
            }
            Accumulator::Unpacked { elements, args } => {
                let values = Value::new_single(Vector::cast_from(elements.item()));
                let indices = match args {
                    Value::None => Value::new_None(),
                    Value::Single(coord) => Value::new_single(Vector::cast_from(coord.unwrap())),
                    Value::Multiple(_) => {
                        panic!("a max accumulator holds at most one coordinate vector")
                    }
                };
                (values, indices)
            }
        }
    }
}

impl<P: ReducePrecision> ReduceWithIndices<P> for Max {}

/// Collapse the vectorized accumulator lanes down to the final maximum and its
/// coordinate, for the parallel layout.
///
/// Ties break towards the lower coordinate, matching the CPU reference. The
/// accumulator must have been built with coordinate tracking on.
#[cube]
fn max_finalize_with_coords<P: ReducePrecision>(
    elements: &Value<Vector<P::EA, P::SI>>,
    args: &Value<Vector<u32, P::SI>>,
) -> (P::EA, u32) {
    let vector_size = elements.item().vector_size().comptime();

    if vector_size > 1 {
        let mut max = max_identity::<P::EA>();
        let mut coordinate = u32::MAX.runtime();

        #[unroll]
        for k in 0..vector_size {
            let acc_element = elements.item().extract(k);
            let acc_coordinate = args.item().extract(k);

            let (selected, selected_coordinate) = select_argmax(
                Vector::<P::EA, Const<1>>::new(max),
                Vector::<u32, Const<1>>::new(coordinate),
                Vector::<P::EA, Const<1>>::new(acc_element),
                Vector::<u32, Const<1>>::new(acc_coordinate),
            );

            max = selected.extract(0usize);
            coordinate = selected_coordinate.extract(0usize);
        }

        (max, coordinate)
    } else {
        (elements.item().extract(0usize), args.item().extract(0usize))
    }
}
