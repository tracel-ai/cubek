use cubecl::comptime;
use cubecl::cube;
use cubecl::frontend::CubeIndexMutExpand;
use cubecl::prelude::*;

use crate::components::instructions::AccumulatorFormat;
use crate::components::instructions::plane_topk_insert;
use crate::components::instructions::plane_topk_merge;
use crate::components::instructions::{Accumulator, Item, Value};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{ReduceRequirements, ReduceStep, SharedAccumulator},
};
use cubecl::frontend::Numeric;

#[derive(Debug, CubeType, Clone)]
pub struct ArgTopK {
    #[cube(comptime)]
    pub k: usize,
}

impl ReduceFamily for ArgTopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = usize;
}

#[derive(CubeType)]
pub struct ArgTopkAccumulator<E: Scalar, S: Size> {
    pub elements: Array<Vector<E, S>>,
    pub coordinates: Array<Vector<u32, S>>,
}

#[derive(CubeType)]
pub struct ArgTopKSharedAccumulator<P: ReducePrecision> {
    elements: Sequence<SharedMemory<Vector<P::EA, P::SI>>>,
    args: Sequence<SharedMemory<Vector<u32, P::SI>>>,
    #[cube(comptime)]
    k: usize,
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P, ArgTopK> for ArgTopKSharedAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &ArgTopK) -> Self {
        let mut elements = Sequence::new();
        let mut args = Sequence::new();
        for _ in 0..inst.k {
            elements.push(SharedMemory::new(length));
            args.push(SharedMemory::new(length));
        }
        ArgTopKSharedAccumulator::<P> {
            elements,
            args,
            k: inst.k,
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let mut values = Array::new(accumulator.k);
        let mut args = Array::new(accumulator.k);
        #[unroll]
        for i in 0..accumulator.k {
            values[i] = accumulator.elements[i][index];
            args[i] = accumulator.args[i][index];
        }
        Accumulator::<P> {
            elements: Value::new_Multiple(values),
            args: Value::new_Multiple(args),
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        let values = item.elements.multiple();
        let args = item.args.multiple();
        #[unroll]
        for i in 0..accumulator.k {
            let values_acc = values[i];
            let args_acc = args[i];

            let mut shared_acc = accumulator.elements[i];
            shared_acc[index] = values_acc;

            let mut shared_arg_acc = accumulator.args[i];
            shared_arg_acc[index] = args_acc;
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ArgTopK {
    type SharedAccumulator = ArgTopKSharedAccumulator<P>;
    type Config = usize;

    fn requirements(_this: &Self) -> super::ReduceRequirements {
        ReduceRequirements { coordinates: true }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Multiple(this.k))
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        ArgTopK { k: config }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::min_value())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let mut elements = Array::new(comptime!(this.k));
        let mut args = Array::new(comptime!(this.k));
        #[unroll]
        for i in 0..this.k {
            elements[i] = Vector::new(P::EA::min_value());
            args[i] = Vector::new(u32::MAX);
        }

        Accumulator::<P> {
            elements: Value::new_Multiple(elements),
            args: Value::new_Multiple(args),
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
                    true,
                );
            }
            ReduceStep::Identity => {
                let coordinates = accumulator.args.multiple_mut();
                let mut insert_val = Vector::cast_from(item.elements);
                let mut insert_coord = item.args.item();

                for j in 0..this.k {
                    let to_keep = select_many(
                        elements[j].equal(insert_val),
                        coordinates[j].less_than(insert_coord),
                        elements[j].greater_than(insert_val),
                    );
                    let best_value = select_many(to_keep, elements[j], insert_val);
                    let loser_value = select_many(to_keep, insert_val, elements[j]);
                    let best_coordinate = select_many(to_keep, coordinates[j], insert_coord);
                    let loser_coordinate = select_many(to_keep, insert_coord, coordinates[j]);

                    elements[j] = best_value;
                    coordinates[j] = best_coordinate;
                    insert_val = loser_value;
                    insert_coord = loser_coordinate;
                }
            }
        };
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        plane_topk_merge::<P::EA, P::SI>(
            accumulator.elements.multiple_mut(),
            &mut accumulator.args,
            this.k,
            true,
        );
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let elements = accumulator.elements.multiple_mut();
        let coordinates = accumulator.args.multiple_mut();
        let other_elements = other.elements.multiple();
        let other_coords = other.args.multiple();

        for i in 0..this.k {
            let mut insert_val = other_elements[i];
            let mut insert_coord = other_coords[i];
            for j in 0..this.k {
                let to_keep = select_many(
                    elements[j].equal(insert_val),
                    coordinates[j].less_than(insert_coord),
                    elements[j].greater_than(insert_val),
                );
                let best_value = select_many(to_keep, elements[j], insert_val);
                let best_coordinate = select_many(to_keep, coordinates[j], insert_coord);
                let loser_value = select_many(to_keep, insert_val, elements[j]);
                let loser_coordinate = select_many(to_keep, insert_coord, coordinates[j]);

                elements[j] = best_value;
                coordinates[j] = best_coordinate;
                insert_val = loser_value;
                insert_coord = loser_coordinate;
            }
        }
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let coords = accumulator.args.multiple();
        let vals = accumulator.elements.multiple();
        let vector_size = coords[0].size().comptime();

        let mut topk_vals = Array::new(this.k);
        let mut topk_coords = Array::new(this.k);

        #[unroll]
        for slot in 0..this.k {
            topk_vals[slot] = P::EA::min_value();
            topk_coords[slot] = u32::MAX;
        }

        #[unroll]
        for i in 0..this.k {
            #[unroll]
            for j in 0..vector_size {
                let mut value = vals[i][j];
                let mut coordinate = coords[i][j];

                #[unroll]
                for slot in 0..this.k {
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

        let mut out = Array::new(this.k);
        #[unroll]
        for i in 0..this.k {
            out[i] = Out::cast_from(topk_coords[i]);
        }
        Value::new_Multiple(out)
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        let acc_args = accumulator.args.multiple();
        let mut output = Array::new(this.k);

        #[unroll]
        for i in 0..this.k {
            output[i] = Vector::cast_from(acc_args[i]);
        }

        Value::new_Multiple(output)
    }
}
