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
pub struct TopK {
    #[cube(comptime)]
    pub k: usize,
}

impl ReduceFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = usize;
}

#[derive(CubeType)]
pub struct TopkAccumulator<E: Scalar, S: Size> {
    pub elements: Array<Vector<E, S>>,
    pub coordinates: Array<Vector<u32, S>>,
}

#[derive(CubeType)]
pub struct TopKSharedAccumulator<P: ReducePrecision> {
    elements: Sequence<SharedMemory<Vector<P::EA, P::SI>>>,
    #[cube(comptime)]
    k: usize,
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P, TopK> for TopKSharedAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &TopK) -> Self {
        let mut elements = Sequence::new();
        for _ in 0..inst.k {
            elements.push(SharedMemory::new(length));
        }
        TopKSharedAccumulator::<P> {
            elements,
            k: inst.k,
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let mut values = Array::new(accumulator.k);
        #[unroll]
        for i in 0..accumulator.k {
            values[i] = accumulator.elements[i][index];
        }
        Accumulator::<P> {
            elements: Value::new_Multiple(values),
            args: Value::new_None(),
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        let values = item.elements.multiple();
        #[unroll]
        for i in 0..accumulator.k {
            let acc = values[i];
            let mut shared_acc = accumulator.elements[i];
            shared_acc[index] = acc;
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for TopK {
    type SharedAccumulator = TopKSharedAccumulator<P>;
    type Config = usize;

    fn requirements(_this: &Self) -> super::ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Multiple(this.k))
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        TopK { k: config }
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

        Accumulator::<P> {
            elements: Value::new_Multiple(elements),
            args: Value::new_None(),
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
                    false,
                );
            }
            ReduceStep::Identity => {
                let mut insert_item = Vector::cast_from(item.elements);

                for j in 0..this.k {
                    let acc_item = elements[j];
                    let keep = acc_item.greater_than(insert_item);

                    elements[j] = select_many(keep, acc_item, insert_item);
                    insert_item = select_many(keep, insert_item, acc_item);
                }
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        plane_topk_merge(
            accumulator.elements.multiple_mut(),
            &mut accumulator.args,
            this.k,
            false,
        );
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let acc_elements = accumulator.elements.multiple_mut();
        let other_elements = other.elements.multiple();

        for i in 0..this.k {
            let mut item = other_elements[i];
            for j in 0..this.k {
                let current_item = acc_elements[j];
                let keep = current_item.greater_than(item);

                let new_top_item = select_many(keep, current_item, item);
                let new_rest_item = select_many(keep, item, current_item);

                acc_elements[j] = new_top_item;
                item = new_rest_item;
            }
        }
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Out> {
        let accumulators = accumulator.elements.multiple();
        let vector_size = accumulators[0].size().comptime();

        let mut topk = Array::new(this.k);
        #[unroll]
        for slot in 0..this.k {
            topk[slot] = Out::min_value();
        }

        #[unroll]
        for i in 0..this.k {
            #[unroll]
            for j in 0..vector_size {
                let mut element = Out::cast_from(accumulators[i][j]);

                #[unroll]
                for slot in 0..this.k {
                    let current = topk[slot];

                    let keep = current > element;

                    topk[slot] = select(keep, current, element);
                    element = select(keep, element, current);
                }
            }
        }

        Value::new_Multiple(topk)
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        let acc_values = accumulator.elements.multiple();
        let mut output = Array::new(this.k);

        #[unroll]
        for i in 0..this.k {
            output[i] = Vector::cast_from(acc_values[i]);
        }

        Value::new_Multiple(output)
    }
}
