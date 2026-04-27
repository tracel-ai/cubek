use super::{
    ArgMax, ArgMin, ArgTopK, Max, MaxAbs, Mean, Min, Prod, ReduceFamily, ReduceInstruction,
    ReduceRequirements, SharedAccumulator, Sum,
};
use crate::components::instructions::{
    Accumulator, AccumulatorFormat, Item, SharedAccumulatorKind, TopK,
};
use crate::{
    ReduceDtypes,
    components::{
        instructions::{ReduceStep, Value},
        precision::ReducePrecision,
    },
};
use cubecl::{
    ir::{ElemType, FloatKind, IntKind, UIntKind},
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, CubeType, Clone)]
pub enum ReduceOperation {
    Sum(Sum),
    Prod(Prod),
    Mean(Mean),
    MaxAbs(MaxAbs),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
    Max(Max),
    Min(Min),
    ArgTopK(ArgTopK),
    TopK(TopK),
}

#[derive_cube_comptime]
#[derive(Serialize, Deserialize)]
pub enum ReduceOperationConfig {
    Sum,
    Prod,
    Mean,
    MaxAbs,
    ArgMax,
    ArgMin,
    Max,
    Min,
    ArgTopK(usize),
    TopK(usize),
}

impl ReduceOperationConfig {
    /// Computes the best case precision for the given config.
    pub fn precision(&self, input: ElemType, output: Option<ElemType>) -> ReduceDtypes {
        match self {
            ReduceOperationConfig::Sum
            | ReduceOperationConfig::Prod
            | ReduceOperationConfig::Mean => {}
            // No benefit to mixed precision accumulation.
            ReduceOperationConfig::MaxAbs
            | ReduceOperationConfig::Max
            | ReduceOperationConfig::Min => {
                return ReduceDtypes {
                    input: input.into(),
                    output: input.into(),
                    accumulation: input.into(),
                };
            }
            ReduceOperationConfig::ArgMax | ReduceOperationConfig::ArgMin => {
                return ReduceDtypes {
                    input: input.into(),
                    output: output
                        .expect("ArgMax and ArgMin must specify output type")
                        .into(),
                    accumulation: input.into(),
                };
            }
            ReduceOperationConfig::ArgTopK(_k) => {
                return ReduceDtypes {
                    input: input.into(),
                    output: output.expect("ArgTopK must specify output type").into(),
                    accumulation: input.into(),
                };
            }
            ReduceOperationConfig::TopK(_k) => {
                return ReduceDtypes {
                    input: input.into(),
                    output: output.expect("TopK must specify output type").into(),
                    accumulation: input.into(),
                };
            }
        };

        match input {
            ElemType::Float(kind) => {
                let acc = match kind {
                    FloatKind::F64 => f64::as_type_native_unchecked(),
                    _ => f32::as_type_native_unchecked(),
                };

                ReduceDtypes {
                    input: input.into(),
                    output: input.into(),
                    accumulation: acc.storage_type(),
                }
            }
            ElemType::Int(kind) => {
                let acc = match kind {
                    IntKind::I64 => i64::as_type_native_unchecked(),
                    _ => i32::as_type_native_unchecked(),
                };

                ReduceDtypes {
                    input: input.into(),
                    output: input.into(),
                    accumulation: acc.storage_type(),
                }
            }
            ElemType::UInt(kind) => {
                let acc = match kind {
                    UIntKind::U64 => u64::as_type_native_unchecked(),
                    _ => u32::as_type_native_unchecked(),
                };

                ReduceDtypes {
                    input: input.into(),
                    output: input.into(),
                    accumulation: acc.storage_type(),
                }
            }
            ElemType::Bool => panic!("Can't reduce on booleans"),
        }
    }
}

impl ReduceFamily for ReduceOperation {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOperationConfig;
}

#[derive(CubeType)]
pub struct DynamicSharedAccumulator<P: ReducePrecision> {
    pub elements: SharedAccumulatorKind<Vector<P::EA, P::SI>>,
    pub args: SharedAccumulatorKind<Vector<u32, P::SI>>,
}

#[derive(CubeType)]
pub struct DynamicAccumulator<P: ReducePrecision> {
    pub elements: Value<Vector<P::EA, P::SI>>,
    pub args: Value<Vector<u32, P::SI>>,
}

#[cube]
impl<P: ReducePrecision, I: ReduceInstruction<P>> SharedAccumulator<P, I>
    for DynamicSharedAccumulator<P>
{
    fn allocate(#[comptime] length: usize, #[comptime] coordinate: bool, inst: &I) -> Self {
        let format = I::accumulator_format(inst);
        match comptime!(format) {
            AccumulatorFormat::Single => {
                let elements = SharedMemory::new(length);
                // TODO how to put multiple?
                let args = if coordinate {
                    let args = SharedMemory::new(length);
                    SharedAccumulatorKind::new_Single(args)
                } else {
                    SharedAccumulatorKind::new_None()
                };
                DynamicSharedAccumulator::<P> {
                    elements: SharedAccumulatorKind::new_Single(elements),
                    args,
                }
            }
            AccumulatorFormat::Multiple(len) => {
                let mut elements = Sequence::new();
                #[unroll]
                for _ in 0..len {
                    elements.push(SharedMemory::new(length));
                }

                if comptime!(!coordinate) {
                    DynamicSharedAccumulator::<P> {
                        elements: SharedAccumulatorKind::new_Multiple(elements),
                        args: SharedAccumulatorKind::new_None(),
                    }
                } else {
                    let mut args = Sequence::new();
                    #[unroll]
                    for _ in 0..len {
                        args.push(SharedMemory::new(length));
                    }
                    DynamicSharedAccumulator::<P> {
                        elements: SharedAccumulatorKind::new_Multiple(elements),
                        args: SharedAccumulatorKind::new_Multiple(args),
                    }
                }
            }
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let elements = accumulator.elements.get(index);
        let args = accumulator.args.get(index);

        Accumulator::<P> { elements, args }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        accumulator.elements.set(index, item.elements);
        accumulator.args.set(index, item.args);
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ReduceOperation {
    type SharedAccumulator = DynamicSharedAccumulator<P>;
    type Config = ReduceOperationConfig;

    fn requirements(this: &Self) -> ReduceRequirements {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::requirements(sum),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::requirements(prod),
            ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::requirements(mean),
            ReduceOperation::MaxAbs(max_abs) => {
                <MaxAbs as ReduceInstruction<P>>::requirements(max_abs)
            }
            ReduceOperation::ArgMax(arg_max) => {
                <ArgMax as ReduceInstruction<P>>::requirements(arg_max)
            }
            ReduceOperation::ArgMin(arg_min) => {
                <ArgMin as ReduceInstruction<P>>::requirements(arg_min)
            }
            ReduceOperation::ArgTopK(arg_topk) => {
                <ArgTopK as ReduceInstruction<P>>::requirements(arg_topk)
            }
            ReduceOperation::TopK(topk) => <TopK as ReduceInstruction<P>>::requirements(topk),
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::requirements(max),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::requirements(min),
        }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::accumulator_format(sum),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::accumulator_format(prod),
            ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::accumulator_format(mean),
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::accumulator_format(maxabs)
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::accumulator_format(argmax)
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::accumulator_format(argmin)
            }
            ReduceOperation::ArgTopK(args) => {
                <ArgTopK as ReduceInstruction<P>>::accumulator_format(args)
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::accumulator_format(max),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::accumulator_format(min),
            ReduceOperation::TopK(topk) => <TopK as ReduceInstruction<P>>::accumulator_format(topk),
        }
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        match config {
            ReduceOperationConfig::Sum => ReduceOperation::new_Sum(Sum {}),
            ReduceOperationConfig::Prod => ReduceOperation::new_Prod(Prod {}),
            ReduceOperationConfig::Mean => ReduceOperation::new_Mean(Mean { sum: Sum {} }),
            ReduceOperationConfig::MaxAbs => ReduceOperation::new_MaxAbs(MaxAbs {}),
            ReduceOperationConfig::ArgMax => ReduceOperation::new_ArgMax(ArgMax {}),
            ReduceOperationConfig::ArgMin => ReduceOperation::new_ArgMin(ArgMin {}),
            ReduceOperationConfig::ArgTopK(k) => ReduceOperation::new_ArgTopK(ArgTopK { k }),
            ReduceOperationConfig::Max => ReduceOperation::new_Max(Max {}),
            ReduceOperationConfig::Min => ReduceOperation::new_Min(Min {}),
            ReduceOperationConfig::TopK(k) => ReduceOperation::new_TopK(TopK { k }),
        }
    }

    fn null_input(this: &Self) -> Vector<P::EI, P::SI> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::null_input(sum),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::null_input(prod),
            ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::null_input(mean),
            ReduceOperation::MaxAbs(maxabs) => <MaxAbs as ReduceInstruction<P>>::null_input(maxabs),
            ReduceOperation::ArgMax(argmax) => <ArgMax as ReduceInstruction<P>>::null_input(argmax),
            ReduceOperation::ArgMin(argmin) => <ArgMin as ReduceInstruction<P>>::null_input(argmin),
            ReduceOperation::ArgTopK(args) => <ArgTopK as ReduceInstruction<P>>::null_input(args),
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::null_input(max),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::null_input(min),
            ReduceOperation::TopK(topk) => <TopK as ReduceInstruction<P>>::null_input(topk),
        }
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::null_accumulator(sum),
            ReduceOperation::Mean(sum) => <Mean as ReduceInstruction<P>>::null_accumulator(sum),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::null_accumulator(prod),
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::null_accumulator(maxabs)
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::null_accumulator(argmax)
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::null_accumulator(argmin)
            }
            ReduceOperation::ArgTopK(args) => {
                <ArgTopK as ReduceInstruction<P>>::null_accumulator(args)
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::null_accumulator(max),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::null_accumulator(min),
            ReduceOperation::TopK(topk) => <TopK as ReduceInstruction<P>>::null_accumulator(topk),
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        match this {
            ReduceOperation::Sum(sum) => {
                <Sum as ReduceInstruction<P>>::reduce(sum, accumulator, item, reduce_step)
            }
            ReduceOperation::Prod(sum) => {
                <Prod as ReduceInstruction<P>>::reduce(sum, accumulator, item, reduce_step)
            }
            ReduceOperation::Mean(sum) => {
                <Mean as ReduceInstruction<P>>::reduce(sum, accumulator, item, reduce_step)
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::reduce(maxabs, accumulator, item, reduce_step)
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::reduce(argmax, accumulator, item, reduce_step)
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::reduce(argmin, accumulator, item, reduce_step)
            }
            ReduceOperation::ArgTopK(argtopk) => {
                <ArgTopK as ReduceInstruction<P>>::reduce(argtopk, accumulator, item, reduce_step)
            }
            ReduceOperation::Max(max) => {
                <Max as ReduceInstruction<P>>::reduce(max, accumulator, item, reduce_step)
            }
            ReduceOperation::Min(min) => {
                <Min as ReduceInstruction<P>>::reduce(min, accumulator, item, reduce_step)
            }
            ReduceOperation::TopK(topk) => {
                <TopK as ReduceInstruction<P>>::reduce(topk, accumulator, item, reduce_step)
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        match this {
            ReduceOperation::Sum(sum) => {
                <Sum as ReduceInstruction<P>>::plane_reduce_inplace(sum, accumulator)
            }
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::plane_reduce_inplace(prod, accumulator)
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::plane_reduce_inplace(mean, accumulator)
            }
            ReduceOperation::MaxAbs(max_abs) => {
                <MaxAbs as ReduceInstruction<P>>::plane_reduce_inplace(max_abs, accumulator)
            }
            ReduceOperation::ArgMax(arg_max) => {
                <ArgMax as ReduceInstruction<P>>::plane_reduce_inplace(arg_max, accumulator)
            }
            ReduceOperation::ArgMin(arg_min) => {
                <ArgMin as ReduceInstruction<P>>::plane_reduce_inplace(arg_min, accumulator)
            }
            ReduceOperation::Max(max) => {
                <Max as ReduceInstruction<P>>::plane_reduce_inplace(max, accumulator)
            }
            ReduceOperation::Min(min) => {
                <Min as ReduceInstruction<P>>::plane_reduce_inplace(min, accumulator)
            }
            ReduceOperation::ArgTopK(argtopk) => {
                <ArgTopK as ReduceInstruction<P>>::plane_reduce_inplace(argtopk, accumulator)
            }
            ReduceOperation::TopK(topk) => {
                <TopK as ReduceInstruction<P>>::plane_reduce_inplace(topk, accumulator)
            }
        }
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        match this {
            ReduceOperation::Sum(sum) => {
                <Sum as ReduceInstruction<P>>::fuse_accumulators(sum, accumulator, other)
            }
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::fuse_accumulators(prod, accumulator, other)
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::fuse_accumulators(mean, accumulator, other)
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::fuse_accumulators(maxabs, accumulator, other)
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::fuse_accumulators(argmax, accumulator, other)
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::fuse_accumulators(argmin, accumulator, other)
            }
            ReduceOperation::ArgTopK(argtopk) => {
                <ArgTopK as ReduceInstruction<P>>::fuse_accumulators(argtopk, accumulator, other)
            }
            ReduceOperation::Max(max) => {
                <Max as ReduceInstruction<P>>::fuse_accumulators(max, accumulator, other)
            }
            ReduceOperation::Min(min) => {
                <Min as ReduceInstruction<P>>::fuse_accumulators(min, accumulator, other)
            }
            ReduceOperation::TopK(topk) => {
                <TopK as ReduceInstruction<P>>::fuse_accumulators(topk, accumulator, other)
            }
        }
    }

    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> Value<Out> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::to_output_parallel::<Out>(
                sum,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    prod,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    mean,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    maxabs,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    argmax,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    argmin,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgTopK(argtopk) => {
                <ArgTopK as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    argtopk,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::to_output_parallel::<Out>(
                max,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::to_output_parallel::<Out>(
                min,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::TopK(topk) => {
                <TopK as ReduceInstruction<P>>::to_output_parallel::<Out>(
                    topk,
                    accumulator,
                    shape_axis_reduce,
                )
            }
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::to_output_perpendicular::<
                Out,
            >(sum, accumulator, shape_axis_reduce),
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    prod,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    mean,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    maxabs,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMax(args) => {
                <ArgMax as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    args,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgTopK(argtopk) => {
                <ArgTopK as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    argtopk,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMin(args) => {
                <ArgMin as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    args,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::to_output_perpendicular::<
                Out,
            >(max, accumulator, shape_axis_reduce),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::to_output_perpendicular::<
                Out,
            >(min, accumulator, shape_axis_reduce),
            ReduceOperation::TopK(topk) => {
                <TopK as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    topk,
                    accumulator,
                    shape_axis_reduce,
                )
            }
        }
    }
}
