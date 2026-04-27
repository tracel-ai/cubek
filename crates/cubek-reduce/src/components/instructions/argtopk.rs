use std::marker::PhantomData;

use cubecl::comptime;
use cubecl::cube;
use cubecl::frontend::CubeIndexMutExpand;
use cubecl::prelude::*;

use crate::components::instructions::AccumulatorFormat;
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
pub struct DummyArgTopkSharedAccumulator<A: CubeType + Send + Sync + 'static> {
    #[cube(comptime)]
    _phantom: PhantomData<A>,
}

#[cube]
impl<A: CubeType + Send + Sync + 'static, P: ReducePrecision> SharedAccumulator<P, ArgTopK>
    for DummyArgTopkSharedAccumulator<A>
{
    fn allocate(
        #[comptime] _length: usize,
        #[comptime] _coordinate: bool,
        _inst: &ArgTopK,
    ) -> Self {
        unreachable!()
        //DummyArgTopkSharedAccumulator {}
    }

    fn read(_accumulator: &Self, _index: usize) -> Accumulator<P> {
        unreachable!()
    }

    fn write(_accumulator: &mut Self, _index: usize, _item: Accumulator<P>) {
        unreachable!()
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ArgTopK {
    type SharedAccumulator = DummyArgTopkSharedAccumulator<Accumulator<P>>;
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
        let coordinate = item.args.item();
        let item = item.elements;

        let (candidate_item, candidate_coordinate) = match reduce_step {
            ReduceStep::Plane => {
                todo!()
            }
            ReduceStep::Identity => (item, coordinate),
        };

        let elements = accumulator.elements.multiple_mut();
        let args = accumulator.args.multiple_mut();
        let mut item = Vector::cast_from(candidate_item);
        let mut coordinate = candidate_coordinate;

        #[unroll]
        for k_iter in 0..this.k {
            let current_item = elements[k_iter];
            let current_coord = args[k_iter];

            // keep "0" means items[0] wins the top slot
            let keep0 = select_many(
                current_item.equal(item),
                current_coord.less_than(coordinate),
                current_item.greater_than(item),
            );

            let new_top_item = select_many(keep0, current_item, item);
            let new_top_coord = select_many(keep0, current_coord, coordinate);
            let new_rest_item = select_many(keep0, item, current_item);
            let new_rest_coord = select_many(keep0, coordinate, current_coord);

            elements[k_iter] = new_top_item;
            args[k_iter] = new_top_coord;
            item = new_rest_item;
            coordinate = new_rest_coord;
        }
    }

    fn plane_reduce_inplace(_this: &Self, _accumulator: &mut Accumulator<P>) {
        todo!()
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let acc_elements = accumulator.elements.multiple_mut();
        let acc_args = accumulator.args.multiple_mut();

        let other_elements = other.elements.multiple();
        let other_args = other.args.multiple();

        for i in 0..this.k {
            let mut item = other_elements[i];
            let mut coordinate = other_args[i];

            for j in 0..this.k {
                let current_item = acc_elements[j];
                let current_coord = acc_args[j];

                let keep0 = select_many(
                    current_item.equal(item),
                    current_coord.less_than(coordinate),
                    current_item.greater_than(item),
                );

                let new_top_item = select_many(keep0, current_item, item);
                let new_top_coord = select_many(keep0, current_coord, coordinate);
                let new_rest_item = select_many(keep0, item, current_item);
                let new_rest_coord = select_many(keep0, coordinate, current_coord);

                acc_elements[j] = new_top_item;
                acc_args[j] = new_top_coord;

                item = new_rest_item;
                coordinate = new_rest_coord;
            }
        }
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let accumulators = accumulator.elements.multiple();
        let args = accumulator.args.multiple();
        let vector_size = accumulators[0].size().comptime();

        let mut topk = Array::new(this.k);
        let mut topk_args = Array::new(this.k);
        #[unroll]
        for slot in 0..this.k {
            topk[slot] = P::EA::min_value();
            topk_args[slot] = Out::cast_from(u32::MAX);
        }

        #[unroll]
        for i in 0..this.k {
            #[unroll]
            for j in 0..vector_size {
                let mut element = accumulators[i][j];
                let mut coord = Out::cast_from(args[i][j]);

                #[unroll]
                for slot in 0..this.k {
                    let current = topk[slot];
                    let current_coord = topk_args[slot];

                    // keep `current` in the slot if it wins (bigger, or equal with lower coord)
                    let keep = select(current == element, current_coord < coord, current > element);

                    topk[slot] = select(keep, current, element);
                    topk_args[slot] = select(keep, current_coord, coord);
                    element = select(keep, element, current);
                    coord = select(keep, coord, current_coord);
                }
            }
        }

        Value::new_Multiple(topk_args)
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
