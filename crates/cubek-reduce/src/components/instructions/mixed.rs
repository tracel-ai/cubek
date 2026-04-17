use super::{
    ArgMax, ArgMin, ArgTopK, Max, MaxAbs, Mean, Min, Prod, ReduceFamily, ReduceInstruction,
    ReduceRequirements, SharedAccumulator, Sum,
};
use crate::components::instructions::{Accumulator, Item, SharedAccumulatorKind};
use crate::{
    ReduceDtypes,
    components::{
        instructions::{AccumulatorKind, ReduceStep},
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
    ArgTopK(u32),
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
    pub elements: AccumulatorKind<Vector<P::EA, P::SI>>,
    pub args: AccumulatorKind<Vector<u32, P::SI>>,
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P> for DynamicSharedAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] coordinate: bool) -> Self {
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
        let coordinates = match this {
            ReduceOperation::Sum(..) => false,
            ReduceOperation::Prod(..) => false,
            ReduceOperation::Mean(..) => false,
            ReduceOperation::MaxAbs(..) => false,
            ReduceOperation::ArgMax(..) => true,
            ReduceOperation::ArgMin(..) => true,
            ReduceOperation::ArgTopK(..) => true,
            ReduceOperation::Max(..) => false,
            ReduceOperation::Min(..) => false,
        };
        ReduceRequirements { coordinates }
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
        }
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        match this {
            ReduceOperation::Sum(sum) => {
                let acc = <Sum as ReduceInstruction<P>>::null_accumulator(sum);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Mean(sum) => {
                let acc = <Mean as ReduceInstruction<P>>::null_accumulator(sum);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Prod(prod) => {
                let acc = <Prod as ReduceInstruction<P>>::null_accumulator(prod);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let acc = <MaxAbs as ReduceInstruction<P>>::null_accumulator(maxabs);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let acc = <ArgMax as ReduceInstruction<P>>::null_accumulator(argmax);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_single(acc.args.item()),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let acc = <ArgMin as ReduceInstruction<P>>::null_accumulator(argmin);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_single(acc.args.item()),
                }
            }
            ReduceOperation::ArgTopK(args) => {
                let topk_accumulator = <ArgTopK as ReduceInstruction<P>>::null_accumulator(args);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_Multiple(topk_accumulator.elements.multiple()),
                    args: AccumulatorKind::new_Multiple(topk_accumulator.args.multiple()),
                }
            }
            ReduceOperation::Max(max) => {
                let acc = <Max as ReduceInstruction<P>>::null_accumulator(max);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let acc = <Min as ReduceInstruction<P>>::null_accumulator(min);

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
        }
    }

    // fn split_accumulator(
    //     this: &Self,
    //     accumulator: &Self::Accumulator,
    // ) -> (
    //     AccumulatorKind<Vector<P::EI, P::SI>>,
    //     ReduceCoordinate<P::SI>,
    // ) {
    //     match this {
    //         ReduceOperation::Sum(sum) => {
    //             <Sum as ReduceInstruction<P>>::split_accumulator(sum, &accumulator.elements.item())
    //         }
    //         ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::split_accumulator(
    //             prod,
    //             &accumulator.elements.item(),
    //         ),
    //         ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::split_accumulator(
    //             mean,
    //             &accumulator.elements.item(),
    //         ),
    //         ReduceOperation::MaxAbs(maxabs) => <MaxAbs as ReduceInstruction<P>>::split_accumulator(
    //             maxabs,
    //             &accumulator.elements.item(),
    //         ),
    //         ReduceOperation::ArgMax(argmax) => <ArgMax as ReduceInstruction<P>>::split_accumulator(
    //             argmax,
    //             &(accumulator.elements.item(), accumulator.args.item()),
    //         ),
    //         ReduceOperation::ArgMin(argmin) => <ArgMin as ReduceInstruction<P>>::split_accumulator(
    //             argmin,
    //             &(accumulator.elements.item(), accumulator.args.item()),
    //         ),
    //         ReduceOperation::ArgTopK(args) => <ArgTopK as ReduceInstruction<P>>::split_accumulator(
    //             args,
    //             // &TopkAccumulator::<P::EA, P::SI> {
    //             //     elements: accumulator.elements.multiple(),
    //             //     coordinates: accumulator.args.multiple(),
    //             // },
    //         ),
    //         ReduceOperation::Max(max) => {
    //             <Max as ReduceInstruction<P>>::split_accumulator(max, &accumulator.elements.item())
    //         }
    //         ReduceOperation::Min(min) => {
    //             <Min as ReduceInstruction<P>>::split_accumulator(min, &accumulator.elements.item())
    //         }
    //     }
    // }

    #[allow(unused_mut)]
    #[allow(unused_mut)]
    fn assign_accumulator(_this: &Self, destination: &mut Accumulator<P>, source: &Accumulator<P>) {
        destination.elements.assign(&source.elements);
        destination.args.assign(&source.args);
    }

    fn reduce(
        this: &Self,
        accumulator: &Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) -> Accumulator<P> {
        match this {
            ReduceOperation::Sum(sum) => {
                let acc =
                    <Sum as ReduceInstruction<P>>::reduce(sum, &accumulator, item, reduce_step);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Prod(sum) => {
                let acc =
                    <Prod as ReduceInstruction<P>>::reduce(sum, &accumulator, item, reduce_step);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Mean(sum) => {
                let acc =
                    <Mean as ReduceInstruction<P>>::reduce(sum, &accumulator, item, reduce_step);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let acc = <MaxAbs as ReduceInstruction<P>>::reduce(
                    maxabs,
                    &accumulator,
                    item,
                    reduce_step,
                );
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let acc = <ArgMax as ReduceInstruction<P>>::reduce(
                    argmax,
                    &accumulator,
                    item,
                    reduce_step,
                );

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_single(acc.args.item()),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let acc = <ArgMin as ReduceInstruction<P>>::reduce(
                    argmin,
                    &accumulator,
                    item,
                    reduce_step,
                );

                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_single(acc.args.item()),
                }
            }
            ReduceOperation::ArgTopK(_args) => {
                todo!()
                // let (elements, args) = <ArgTopK as ReduceInstruction<P>>::reduce(
                //     args,
                //     &(accumulator.elements.array(), accumulator.args.array()),
                //     item,
                //     coordinate,
                //     use_planes,
                // );

                // DynamicAccumulatorItem::<P::EA, P::SI> {
                //     elements,
                //     args: ComptimeOption::new_Some(args),
                // }
            }
            ReduceOperation::Max(max) => {
                let acc =
                    <Max as ReduceInstruction<P>>::reduce(max, &accumulator, item, reduce_step);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let acc =
                    <Min as ReduceInstruction<P>>::reduce(min, &accumulator, item, reduce_step);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
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
            ReduceOperation::ArgTopK(arg_top_k) => {
                <ArgTopK as ReduceInstruction<P>>::plane_reduce_inplace(arg_top_k, accumulator)
            }
        }
    }

    fn fuse_accumulators(
        this: &Self,
        lhs: &Accumulator<P>,
        rhs: &Accumulator<P>,
    ) -> Accumulator<P> {
        match this {
            ReduceOperation::Sum(sum) => {
                let acc = <Sum as ReduceInstruction<P>>::fuse_accumulators(sum, lhs, rhs);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Prod(prod) => {
                let acc = <Prod as ReduceInstruction<P>>::fuse_accumulators(prod, lhs, rhs);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::Mean(mean) => {
                let acc = <Mean as ReduceInstruction<P>>::fuse_accumulators(mean, lhs, rhs);
                Accumulator::<P> {
                    elements: AccumulatorKind::new_single(acc.elements.item()),
                    args: AccumulatorKind::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::fuse_accumulators(maxabs, lhs, rhs)
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::fuse_accumulators(argmax, lhs, rhs)
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::fuse_accumulators(argmin, lhs, rhs)
            }
            ReduceOperation::ArgTopK(_args) => {
                todo!()
                // let acc = <ArgTopK as ReduceInstruction<P>>::fuse_accumulators(
                //     args,
                //     (lhs.elements.array(), lhs.args.array()),
                //     (rhs.elements.array(), rhs.args.array()),
                // );
                // DynamicAccumulatorItem::<P::EA, P::SI> {
                //     elements: AccumulatorKind::new_Array(acc.elements),
                //     args: AccumulatorKind::new_Array(acc.coordinates),
                // }
            }
            ReduceOperation::Max(max) => {
                <Max as ReduceInstruction<P>>::fuse_accumulators(max, lhs, rhs)
            }
            ReduceOperation::Min(min) => {
                <Min as ReduceInstruction<P>>::fuse_accumulators(min, lhs, rhs)
            }
        }
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Dynamic can be implemented by fusing a Sum reduction and a element-wise division.
    fn merge_vector<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::merge_vector::<Out>(
                sum,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::merge_vector::<Out>(
                prod,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::merge_vector::<Out>(
                mean,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::merge_vector::<Out>(
                    maxabs,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::merge_vector::<Out>(
                    argmax,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::merge_vector::<Out>(
                    argmin,
                    accumulator,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgTopK(_args) => {
                todo!()
                // <ArgTopK as ReduceInstruction<P>>::merge_vector::<Out>(
                //     args,
                //     (accumulator.elements.array(), accumulator.args.array()),
                //     shape_axis_reduce,
                // )
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::merge_vector::<Out>(
                max,
                accumulator,
                shape_axis_reduce,
            ),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::merge_vector::<Out>(
                min,
                accumulator,
                shape_axis_reduce,
            ),
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
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
            ReduceOperation::ArgTopK(_args) => {
                todo!()
                // <ArgTopK as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                //     args,
                //     (accumulator.elements, accumulator.args.unwrap()),
                //     shape_axis_reduce,
                // )
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
        }
    }
}
