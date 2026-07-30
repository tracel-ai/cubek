use cubecl::comptime;
use cubecl::cube;
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use crate::components::instructions::AccumulatorFormat;
use crate::components::instructions::plane_topk_insert;
use crate::components::instructions::plane_topk_merge;
use crate::components::instructions::{Accumulator, Item, Value, ValueExpand};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{
        ReduceOutputMode, ReduceRequirements, ReduceStep, ReduceWithIndices,
        ReduceWithIndicesFamily, SharedAccumulator,
    },
};
use cubecl::frontend::Numeric;

#[derive_cube_comptime]
#[derive(Serialize, Deserialize)]
pub struct TopKConfig {
    pub k: usize,
    pub output: ReduceOutputMode,
}

#[derive(Debug, CubeType, Clone)]
pub struct TopK {
    #[cube(comptime)]
    pub k: usize,
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = TopKConfig;
}

impl ReduceWithIndicesFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = TopKConfig;
}

/// Insert `insert_val` into the descending-sorted `elements` (and its
/// coordinate, when it carries one), pushing the smallest slot out.
///
/// Ties break towards the lower coordinate, matching the CPU reference. A
/// coordinate-less candidate emits no index arithmetic at all.
#[cube]
pub(crate) fn topk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    insert_val: Vector<N, S>,
    insert_coord: &Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    let mut insert_val = insert_val;

    match insert_coord {
        Value::None => {
            for j in 0..k {
                let to_keep = elements[j].greater_than(&insert_val);
                let next_val = select_many(to_keep, insert_val, elements[j]);
                elements[j] = select_many(to_keep, elements[j], insert_val);
                insert_val = next_val;
            }
        }
        Value::Single(coord) => {
            let mut insert_coord = coord.unwrap();
            let coords = coordinates.multiple_mut();

            for j in 0..k {
                let to_keep = select_many(
                    elements[j].equal(&insert_val),
                    coords[j].less_than(&insert_coord),
                    elements[j].greater_than(&insert_val),
                );

                let next_val = select_many(to_keep, insert_val, elements[j]);
                elements[j] = select_many(to_keep, elements[j], insert_val);
                insert_val = next_val;

                let next_coord = select_many(to_keep, insert_coord, coords[j]);
                coords[j] = select_many(to_keep, coords[j], insert_coord);
                insert_coord = next_coord;
            }
        }
        Value::Multiple(_) => panic!("a top-k candidate carries at most one coordinate"),
    }
}

#[derive(CubeType)]
pub struct TopKSharedAccumulator<P: ReducePrecision> {
    elements: Sequence<Shared<[Vector<P::EA, P::SI>]>>,
    /// Empty unless the instruction tracks coordinates; its length is the single
    /// source of truth for whether coordinates are staged (see `read`/`write`).
    args: Sequence<Shared<[Vector<u32, P::SI>]>>,
    #[cube(comptime)]
    k: usize,
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P, TopK> for TopKSharedAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &TopK) -> Self {
        let has_coords = comptime!(inst.output.has_indices());

        // Both loops must be unrolled: a `Sequence` is built at expand time, so a
        // runtime loop would run the body once and leave a single slice behind
        // whatever `k` is, and `read`/`write` would then index past the end.
        let mut elements = Sequence::new();
        #[unroll]
        for _ in 0..inst.k {
            elements.push(Shared::new_slice(length));
        }

        let mut args = Sequence::new();
        if has_coords {
            #[unroll]
            for _ in 0..inst.k {
                args.push(Shared::new_slice(length));
            }
        }

        TopKSharedAccumulator::<P> {
            elements,
            args,
            k: inst.k,
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let mut values = Array::new(accumulator.k);
        #[unroll]
        for i in 0..accumulator.k {
            values[i] = accumulator.elements[i][index];
        }

        let num_args = comptime!(accumulator.args.len());
        let args = if comptime!(num_args != 0) {
            let mut args = Array::new(accumulator.k);
            #[unroll]
            for i in 0..accumulator.k {
                args[i] = accumulator.args[i][index];
            }
            Value::new_Multiple(args)
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_Multiple(values),
            args,
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        let values = item.elements.multiple();
        #[unroll]
        for i in 0..accumulator.k {
            let acc = values[i];
            let shared_acc = &mut accumulator.elements[i];
            shared_acc[index] = acc;
        }

        let num_args = comptime!(accumulator.args.len());
        if comptime!(num_args != 0) {
            let args = item.args.multiple();
            #[unroll]
            for i in 0..accumulator.k {
                let arg = args[i];
                let shared_arg_acc = &mut accumulator.args[i];
                shared_arg_acc[index] = arg;
            }
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for TopK {
    type SharedAccumulator = TopKSharedAccumulator<P>;
    type Config = TopKConfig;

    fn requirements(this: &Self) -> super::ReduceRequirements {
        ReduceRequirements {
            coordinates: comptime!(this.output.has_indices()),
        }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Multiple(this.k))
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        TopK {
            k: config.k,
            output: config.output,
        }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::min_value())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let mut elements = Array::new(comptime!(this.k));
        #[unroll]
        for i in 0..this.k {
            elements[i] = Vector::new(P::EA::min_value());
        }

        let args = if comptime!(this.output.has_indices()) {
            let mut args = Array::new(comptime!(this.k));
            #[unroll]
            for i in 0..this.k {
                args[i] = Vector::new(u32::MAX);
            }
            Value::new_Multiple(args)
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_Multiple(elements),
            args,
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let elements = accumulator.elements.multiple_mut();

        match reduce_step {
            ReduceStep::Plane => {
                plane_topk_insert::<P::EA, P::SI>(
                    elements,
                    &mut accumulator.args,
                    Vector::cast_from(item.elements),
                    &item.args,
                    this.k,
                );
            }
            ReduceStep::Identity => {
                topk_insert::<P::EA, P::SI>(
                    elements,
                    &mut accumulator.args,
                    Vector::cast_from(item.elements),
                    &item.args,
                    this.k,
                );
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        plane_topk_merge::<P::EA, P::SI>(
            accumulator.elements.multiple_mut(),
            &mut accumulator.args,
            this.k,
        );
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let elements = accumulator.elements.multiple_mut();
        let other_elements = other.elements.multiple();

        for i in 0..this.k {
            topk_insert::<P::EA, P::SI>(
                elements,
                &mut accumulator.args,
                other_elements[i],
                &other.args.slot(i),
                this.k,
            );
        }
    }

    fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode) {
        comptime!(this.output)
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        match accumulator.args {
            Value::None => {
                let values = topk_finalize_values::<P, Out>(&accumulator, this.k);
                (Value::new_Multiple(values), Value::new_None())
            }
            Value::Multiple(_) => {
                let (values, coords) = topk_finalize_with_coords::<P>(&accumulator, this.k);

                let mut out_values = Array::new(this.k);
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_values[i] = Out::cast_from(values[i]);
                    out_indices[i] = Idx::cast_from(coords[i]);
                }

                (
                    Value::new_Multiple(out_values),
                    Value::new_Multiple(out_indices),
                )
            }
            Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
        }
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        let acc_values = accumulator.elements.multiple();
        let mut out_values = Array::new(this.k);
        #[unroll]
        for i in 0..this.k {
            out_values[i] = Vector::cast_from(acc_values[i]);
        }

        let indices = match &accumulator.args {
            Value::None => Value::new_None(),
            Value::Multiple(acc_args) => {
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_indices[i] = Vector::cast_from(acc_args[i]);
                }
                Value::new_Multiple(out_indices)
            }
            Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
        };

        (Value::new_Multiple(out_values), indices)
    }
}

impl<P: ReducePrecision> ReduceWithIndices<P> for TopK {}

/// Collapse the `k * vector_size` accumulator candidates down to the final `k`
/// values, for the parallel (reduce axis is the vectorized axis) layout.
///
/// Coordinates are not tracked, so ties are broken arbitrarily. Use
/// [`topk_finalize_with_coords`] when indices are wanted.
#[cube]
fn topk_finalize_values<P: ReducePrecision, Out: Numeric>(
    accumulator: &Accumulator<P>,
    #[comptime] k: usize,
) -> Array<Out> {
    let vals = accumulator.elements.multiple();
    let vector_size = vals[0].size().comptime();

    let mut topk = Array::new(k);
    #[unroll]
    for slot in 0..k {
        topk[slot] = Out::min_value();
    }

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut element = Out::cast_from(vals[i].extract(j));

            #[unroll]
            for slot in 0..k {
                let current = topk[slot];
                let keep = current > element;

                topk[slot] = select(keep, current, element);
                element = select(keep, element, current);
            }
        }
    }

    topk
}

/// Collapse the `k * vector_size` accumulator candidates down to the final `k`
/// values *and* their coordinates, for the parallel layout.
///
/// Ties break towards the lower coordinate, matching the CPU reference. The
/// accumulator must have been built with coordinate tracking on.
#[cube]
fn topk_finalize_with_coords<P: ReducePrecision>(
    accumulator: &Accumulator<P>,
    #[comptime] k: usize,
) -> (Array<P::EA>, Array<u32>) {
    let vals = accumulator.elements.multiple();
    let coords = accumulator.args.multiple();
    let vector_size = coords[0].size().comptime();

    let mut topk_vals = Array::new(k);
    let mut topk_coords = Array::new(k);

    #[unroll]
    for slot in 0..k {
        topk_vals[slot] = P::EA::min_value();
        topk_coords[slot] = u32::MAX;
    }

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut value = vals[i].extract(j);
            let mut coordinate = coords[i].extract(j);

            #[unroll]
            for slot in 0..k {
                let current_value = topk_vals[slot];
                let current_coordinate = topk_coords[slot];

                let to_keep = select(
                    current_value == value,
                    current_coordinate < coordinate,
                    current_value > value,
                );

                topk_vals[slot] = select(to_keep, current_value, value);
                topk_coords[slot] = select(to_keep, current_coordinate, coordinate);

                value = select(to_keep, value, current_value);
                coordinate = select(to_keep, coordinate, current_coordinate);
            }
        }
    }

    (topk_vals, topk_coords)
}
