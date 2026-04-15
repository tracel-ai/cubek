use super::{
    ArgMax, ArgMin, ArgTopK, Max, MaxAbs, Mean, Min, Prod, ReduceCoordinate, ReduceFamily,
    ReduceInstruction, ReduceRequirements, SharedAccumulator, Sum,
};
use crate::{
    ReduceDtypes,
    components::{instructions::ReduceStep, precision::ReducePrecision},
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
pub struct DynamicAccumulator<T: Numeric, N: Size> {
    pub elements: SharedMemory<Vector<T, N>>,
    pub args: ComptimeOption<SharedMemory<Vector<u32, N>>>,
}

#[derive(CubeType)]
pub struct DynamicAccumulatorItem<T: Numeric, N: Size> {
    pub elements: Vector<T, N>,
    pub args: ComptimeOption<Vector<u32, N>>,
}

#[cube]
impl<In: Numeric, N: Size> SharedAccumulator for DynamicAccumulator<In, N> {
    type Item = DynamicAccumulatorItem<In, N>;

    fn allocate(#[comptime] length: usize, #[comptime] coordinate: bool) -> Self {
        let elements = SharedMemory::new(length);
        let args = if coordinate {
            let args = SharedMemory::new(length);
            ComptimeOption::new_Some(args)
        } else {
            ComptimeOption::new_None()
        };

        DynamicAccumulator::<In, N> { elements, args }
    }

    fn read(accumulator: &Self, index: usize) -> Self::Item {
        let elements = accumulator.elements[index];
        let args = accumulator.args.map(|args| args[index]);

        DynamicAccumulatorItem::<In, N> { elements, args }
    }

    fn write(accumulator: &mut Self, index: usize, item: Self::Item) {
        accumulator.elements[index] = item.elements;

        let args = &mut accumulator.args;
        #[comptime]
        if let ComptimeOption::Some((args, item_args)) = args.as_mut().zip(item.args) {
            args[index] = item_args;
        };
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ReduceOperation {
    type AccumulatorItem = DynamicAccumulatorItem<P::EA, P::SI>;
    type SharedAccumulator = DynamicAccumulator<P::EA, P::SI>;
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

    fn null_accumulator(this: &Self) -> Self::AccumulatorItem {
        match this {
            ReduceOperation::Sum(sum) => {
                let elements = <Sum as ReduceInstruction<P>>::null_accumulator(sum);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Mean(sum) => {
                let elements = <Mean as ReduceInstruction<P>>::null_accumulator(sum);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Prod(sum) => {
                let elements = <Prod as ReduceInstruction<P>>::null_accumulator(sum);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let elements = <MaxAbs as ReduceInstruction<P>>::null_accumulator(maxabs);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let (elements, args) = <ArgMax as ReduceInstruction<P>>::null_accumulator(argmax);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let (elements, args) = <ArgMin as ReduceInstruction<P>>::null_accumulator(argmin);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgTopK(args) => {
                let (elements, args) = <ArgTopK as ReduceInstruction<P>>::null_accumulator(args);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::Max(max) => {
                let elements = <Max as ReduceInstruction<P>>::null_accumulator(max);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let elements = <Min as ReduceInstruction<P>>::null_accumulator(min);

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
        }
    }

    fn read_accumulator(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        match this {
            ReduceOperation::Sum(sum) => {
                <Sum as ReduceInstruction<P>>::read_accumulator(sum, &accumulator.elements)
            }
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::read_accumulator(prod, &accumulator.elements)
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::read_accumulator(mean, &accumulator.elements)
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::read_accumulator(maxabs, &accumulator.elements)
            }
            ReduceOperation::ArgMax(argmax) => <ArgMax as ReduceInstruction<P>>::read_accumulator(
                argmax,
                &(accumulator.elements, accumulator.args.unwrap()),
            ),
            ReduceOperation::ArgMin(argmin) => <ArgMin as ReduceInstruction<P>>::read_accumulator(
                argmin,
                &(accumulator.elements, accumulator.args.unwrap()),
            ),
            ReduceOperation::ArgTopK(args) => <ArgTopK as ReduceInstruction<P>>::read_accumulator(
                args,
                &(accumulator.elements, accumulator.args.unwrap()),
            ),
            ReduceOperation::Max(max) => {
                <Max as ReduceInstruction<P>>::read_accumulator(max, &accumulator.elements)
            }
            ReduceOperation::Min(min) => {
                <Min as ReduceInstruction<P>>::read_accumulator(min, &accumulator.elements)
            }
        }
    }

    #[allow(unused_mut)]
    #[allow(unused_mut)]
    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        destination.elements = source.elements;
        let args = &mut destination.args;
        #[comptime]
        if let ComptimeOption::Some((mut val, source_val)) = args.as_mut().zip(source.args) {
            *val = source_val;
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Vector<P::EI, P::SI>,
        coordinate: ReduceCoordinate<P::SI>,
        #[comptime] reduce_step: ReduceStep,
    ) -> Self::AccumulatorItem {
        match this {
            ReduceOperation::Sum(sum) => {
                let elements = <Sum as ReduceInstruction<P>>::reduce(
                    sum,
                    &accumulator.elements,
                    item,
                    coordinate,
                    reduce_step,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Prod(sum) => {
                let elements = <Prod as ReduceInstruction<P>>::reduce(
                    sum,
                    &accumulator.elements,
                    item,
                    coordinate,
                    reduce_step,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Mean(sum) => {
                let elements = <Mean as ReduceInstruction<P>>::reduce(
                    sum,
                    &accumulator.elements,
                    item,
                    coordinate,
                    reduce_step,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let elements = <MaxAbs as ReduceInstruction<P>>::reduce(
                    maxabs,
                    &accumulator.elements,
                    item,
                    coordinate,
                    reduce_step,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let (elements, args) = <ArgMax as ReduceInstruction<P>>::reduce(
                    argmax,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    reduce_step,
                );

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let (elements, args) = <ArgMin as ReduceInstruction<P>>::reduce(
                    argmin,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    reduce_step,
                );

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgTopK(args) => {
                let (elements, args) = <ArgTopK as ReduceInstruction<P>>::reduce(
                    args,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    reduce_step,
                );

                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::Max(max) => {
                let elements = <Max as ReduceInstruction<P>>::reduce(
                    max,
                    &accumulator.elements,
                    item,
                    coordinate,
                    reduce_step,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let elements = <Min as ReduceInstruction<P>>::reduce(
                    min,
                    &accumulator.elements,
                    item,
                    coordinate,
                    reduce_step,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
        }
    }

    fn fuse_accumulators(
        this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        match this {
            ReduceOperation::Sum(sum) => {
                let elements = <Sum as ReduceInstruction<P>>::fuse_accumulators(
                    sum,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Prod(prod) => {
                let elements = <Prod as ReduceInstruction<P>>::fuse_accumulators(
                    prod,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Mean(mean) => {
                let elements = <Mean as ReduceInstruction<P>>::fuse_accumulators(
                    mean,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let elements = <MaxAbs as ReduceInstruction<P>>::fuse_accumulators(
                    maxabs,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let (elements, args) = <ArgMax as ReduceInstruction<P>>::fuse_accumulators(
                    argmax,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let (elements, args) = <ArgMin as ReduceInstruction<P>>::fuse_accumulators(
                    argmin,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgTopK(args) => {
                let (elements, args) = <ArgTopK as ReduceInstruction<P>>::fuse_accumulators(
                    args,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_Some(args),
                }
            }
            ReduceOperation::Max(max) => {
                let elements = <Max as ReduceInstruction<P>>::fuse_accumulators(
                    max,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let elements = <Min as ReduceInstruction<P>>::fuse_accumulators(
                    min,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA, P::SI> {
                    elements,
                    args: ComptimeOption::new_None(),
                }
            }
        }
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Dynamic can be implemented by fusing a Sum reduction and a element-wise division.
    fn merge_vector<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: usize,
    ) -> Out {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::merge_vector::<Out>(
                sum,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::merge_vector::<Out>(
                prod,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::merge_vector::<Out>(
                mean,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::merge_vector::<Out>(
                    maxabs,
                    accumulator.elements,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::merge_vector::<Out>(
                    argmax,
                    (accumulator.elements, accumulator.args.unwrap()),
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::merge_vector::<Out>(
                    argmin,
                    (accumulator.elements, accumulator.args.unwrap()),
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgTopK(args) => {
                <ArgTopK as ReduceInstruction<P>>::merge_vector::<Out>(
                    args,
                    (accumulator.elements, accumulator.args.unwrap()),
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::merge_vector::<Out>(
                max,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::merge_vector::<Out>(
                min,
                accumulator.elements,
                shape_axis_reduce,
            ),
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: usize,
    ) -> Vector<Out, P::SI> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::to_output_perpendicular::<
                Out,
            >(sum, accumulator.elements, shape_axis_reduce),
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    prod,
                    accumulator.elements,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    mean,
                    accumulator.elements,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    maxabs,
                    accumulator.elements,
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMax(args) => {
                <ArgMax as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    args,
                    (accumulator.elements, accumulator.args.unwrap()),
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgTopK(args) => {
                <ArgTopK as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    args,
                    (accumulator.elements, accumulator.args.unwrap()),
                    shape_axis_reduce,
                )
            }
            ReduceOperation::ArgMin(args) => {
                <ArgMin as ReduceInstruction<P>>::to_output_perpendicular::<Out>(
                    args,
                    (accumulator.elements, accumulator.args.unwrap()),
                    shape_axis_reduce,
                )
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::to_output_perpendicular::<
                Out,
            >(max, accumulator.elements, shape_axis_reduce),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::to_output_perpendicular::<
                Out,
            >(min, accumulator.elements, shape_axis_reduce),
        }
    }
}
