use cubecl::comptime;
use cubecl::cube;
use cubecl::frontend::CubeIndexMutExpand;
use cubecl::prelude::*;

use crate::components::instructions::AccumulatorFormat;
use crate::components::instructions::lowest_coordinate_matching;
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
/// Only to respect the type system. Shared Accumulator behaviour is not supported
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
impl ArgTopK {
    pub fn choose_best<T: Numeric, N: Size>(
        values0: Vector<T, N>,
        coordinates0: Vector<u32, N>,
        values1: Vector<T, N>,
        coordinates1: Vector<u32, N>,
    ) -> (Vector<T, N>, Vector<u32, N>) {
        let to_keep = select_many(
            values0.equal(values1),
            coordinates0.less_than(coordinates1),
            values0.greater_than(values1),
        );

        let values = select_many(to_keep, values0, values1);
        let coordinates = select_many(to_keep, coordinates0, coordinates1);
        (values, coordinates)
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
        //let coordinate = item.args.item();
        //let item = item.elements;

        let elements = accumulator.elements.multiple_mut();
        let coordinates = accumulator.args.multiple_mut();

        match reduce_step {
            ReduceStep::Plane => {
                plane_argtopk_insert::<P::EA, P::SI>(
                    elements,
                    coordinates,
                    Vector::cast_from(item.elements),
                    item.args.item(),
                    this.k,
                );
            }
            ReduceStep::Identity => {
                let mut insert_val = Vector::cast_from(item.elements);
                let mut insert_coord = item.args.item();

                for j in 0..this.k {
                    let (best_v, best_c) =
                        Self::choose_best(elements[j], coordinates[j], insert_val, insert_coord);

                    let (loser_v, loser_c) =
                        Self::choose_best(insert_val, insert_coord, elements[j], coordinates[j]);

                    elements[j] = best_v;
                    coordinates[j] = best_c;
                    insert_val = loser_v;
                    insert_coord = loser_c;
                }
            }
        };
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        plane_argtopk_merge::<P::EA, P::SI>(
            this.k,
            accumulator.elements.multiple_mut(),
            accumulator.args.multiple_mut(),
        );
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let acc_elements = accumulator.elements.multiple_mut();
        let acc_coords = accumulator.args.multiple_mut();
        let other_elements = other.elements.multiple();
        let other_coords = other.args.multiple();

        for i in 0..this.k {
            let mut val = other_elements[i];
            let mut coord = other_coords[i];
            for j in 0..this.k {
                let (best_v, best_c) =
                    Self::choose_best(acc_elements[j], acc_coords[j], val, coord);
                let (loser_v, loser_c) =
                    Self::choose_best(val, coord, acc_elements[j], acc_coords[j]);

                acc_elements[j] = best_v;
                acc_coords[j] = best_c;
                val = loser_v;
                coord = loser_c;
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

#[cube]
pub fn plane_argtopk_merge<N: Numeric, S: Size>(
    #[comptime] k: usize,
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
) {
    let mut final_elements = Array::new(k);
    let mut final_coords = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let lane_id = Vector::new(UNIT_POS_X);

    for i in 0..k {
        let mut local_val = Vector::new(N::min_value());
        let mut local_coord = Vector::new(u32::MAX);

        for j in 0..k {
            let is_pointed = cursor.equal(Vector::new(j as u32));
            local_val = select_many(is_pointed, elements[j], local_val);
            local_coord = select_many(is_pointed, coordinates[j], local_coord);
        }

        let winning_val = plane_max(local_val);
        let winning_coord = lowest_coordinate_matching(winning_val, local_val, local_coord);

        final_elements[i] = winning_val;
        final_coords[i] = winning_coord;

        let is_candidate = local_val
            .equal(winning_val)
            .and(local_coord.equal(winning_coord));
        let candidate_id = select_many(is_candidate, lane_id, Vector::new(u32::MAX));
        let winning_lane = plane_min(candidate_id);

        let is_winner_thread = lane_id.equal(winning_lane);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    for i in 0..k {
        elements[i] = final_elements[i];
        coordinates[i] = final_coords[i];
    }
}

#[cube]
pub fn plane_argtopk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
    item: Vector<N, S>,
    coord: Vector<u32, S>,
    #[comptime] k: usize,
) {
    let mut local_best_val = item;
    let mut local_best_coord = coord;

    #[unroll]
    for _i in 0..k {
        let winning_val = plane_max(local_best_val);
        let winning_coord =
            lowest_coordinate_matching(winning_val, local_best_val, local_best_coord);

        let mut insert_val = winning_val;
        let mut insert_coord = winning_coord;

        for j in 0..k {
            let (best_v, best_c) =
                ArgTopK::choose_best(elements[j], coordinates[j], insert_val, insert_coord);
            let (loser_v, loser_c) =
                ArgTopK::choose_best(insert_val, insert_coord, elements[j], coordinates[j]);

            elements[j] = best_v;
            coordinates[j] = best_c;
            insert_val = loser_v;
            insert_coord = loser_c;
        }

        // Only the thread that provided the specific pair "wins" and masks it out
        let is_winner = local_best_val
            .equal(winning_val)
            .and(local_best_coord.equal(winning_coord));
        local_best_val = select_many(is_winner, Vector::new(N::min_value()), local_best_val);
        local_best_coord = select_many(is_winner, Vector::new(u32::MAX), local_best_coord);
    }
}
