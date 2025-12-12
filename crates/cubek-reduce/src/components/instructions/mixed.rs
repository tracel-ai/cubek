use super::{
    ArgMax, ArgMin, Max, MaxAbs, Mean, Min, Prod, ReduceCoordinate, ReduceFamily,
    ReduceInstruction, ReduceRequirements, SharedAccumulator, Sum,
};
use crate::{ReduceDtypes, components::precision::ReducePrecision};
use cubecl::{
    ir::{ElemType, FloatKind, IntKind, UIntKind},
    prelude::*,
    std::{CubeOption, CubeOptionExpand},
};

#[derive(Debug, CubeType, Clone)]
#[allow(unused)]
pub(crate) enum ReduceOperation {
    Sum(Sum),
    Prod(Prod),
    Mean(Mean),
    MaxAbs(MaxAbs),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
    Max(Max),
    Min(Min),
}

#[derive_cube_comptime]
pub enum ReduceOperationConfig {
    Sum,
    Prod,
    Mean,
    MaxAbs,
    ArgMax,
    ArgMin,
    Max,
    Min,
}

impl ReduceOperationConfig {
    /// Computes the best case precision for the given config.
    pub fn precision(&self, input: ElemType) -> ReduceDtypes {
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
                    output: i32::as_type_native_unchecked(),
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
                    accumulation: acc,
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
                    accumulation: acc,
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
                    accumulation: acc,
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
pub struct DynamicAccumulator<N: Numeric> {
    pub elements: SharedMemory<Line<N>>,
    pub args: CubeOption<SharedMemory<Line<u32>>>,
}

#[derive(CubeType)]
pub struct DynamicAccumulatorItem<N: Numeric> {
    pub elements: Line<N>,
    pub args: CubeOption<Line<u32>>,
}

#[cube]
impl<In: Numeric> SharedAccumulator for DynamicAccumulator<In> {
    type Item = DynamicAccumulatorItem<In>;

    fn allocate(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
        #[comptime] coordinate: bool,
    ) -> Self {
        let elements = SharedMemory::new_lined(length, line_size);
        let args = if comptime![coordinate] {
            let args = SharedMemory::new_lined(length, line_size);
            CubeOption::new_Some(args)
        } else {
            CubeOption::new_None()
        };

        DynamicAccumulator::<In> { elements, args }
    }

    fn read(accumulator: &Self, index: u32) -> Self::Item {
        let elements = accumulator.elements[index];
        let args = match accumulator.args {
            CubeOption::Some(args) => CubeOption::new_Some(args[index]),
            CubeOption::None => CubeOption::new_None(),
        };

        DynamicAccumulatorItem::<In> { elements, args }
    }

    fn write(accumulator: &mut Self, index: u32, item: Self::Item) {
        accumulator.elements[index] = item.elements;

        let args = &mut accumulator.args;
        match args {
            CubeOption::Some(args) => {
                args[index] = item.args.unwrap();
            }
            CubeOption::None => {}
        };
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ReduceOperation {
    type AccumulatorItem = DynamicAccumulatorItem<P::EA>;
    type SharedAccumulator = DynamicAccumulator<P::EA>;
    type Config = ReduceOperationConfig;

    fn requirements(this: &Self) -> ReduceRequirements {
        let coordinates = match this {
            ReduceOperation::Sum(..) => comptime![false],
            ReduceOperation::Prod(..) => comptime![false],
            ReduceOperation::Mean(..) => comptime![false],
            ReduceOperation::MaxAbs(..) => comptime![false],
            ReduceOperation::ArgMax(..) => comptime![true],
            ReduceOperation::ArgMin(..) => comptime![true],
            ReduceOperation::Max(..) => comptime![false],
            ReduceOperation::Min(..) => comptime![false],
        };
        ReduceRequirements {
            coordinates: comptime! {coordinates},
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
            ReduceOperationConfig::Max => ReduceOperation::new_Max(Max {}),
            ReduceOperationConfig::Min => ReduceOperation::new_Min(Min {}),
        }
    }

    fn null_input(this: &Self, #[comptime] line_size: u32) -> Line<P::EI> {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::null_input(sum, line_size),
            ReduceOperation::Prod(prod) => {
                <Prod as ReduceInstruction<P>>::null_input(prod, line_size)
            }
            ReduceOperation::Mean(mean) => {
                <Mean as ReduceInstruction<P>>::null_input(mean, line_size)
            }
            ReduceOperation::MaxAbs(maxabs) => {
                <MaxAbs as ReduceInstruction<P>>::null_input(maxabs, line_size)
            }
            ReduceOperation::ArgMax(argmax) => {
                <ArgMax as ReduceInstruction<P>>::null_input(argmax, line_size)
            }
            ReduceOperation::ArgMin(argmin) => {
                <ArgMin as ReduceInstruction<P>>::null_input(argmin, line_size)
            }
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::null_input(max, line_size),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::null_input(min, line_size),
        }
    }

    fn null_accumulator(this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        match this {
            ReduceOperation::Sum(sum) => {
                let elements = <Sum as ReduceInstruction<P>>::null_accumulator(sum, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Mean(sum) => {
                let elements = <Mean as ReduceInstruction<P>>::null_accumulator(sum, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Prod(sum) => {
                let elements = <Prod as ReduceInstruction<P>>::null_accumulator(sum, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let elements =
                    <MaxAbs as ReduceInstruction<P>>::null_accumulator(maxabs, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let (elements, args) =
                    <ArgMax as ReduceInstruction<P>>::null_accumulator(argmax, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let (elements, args) =
                    <ArgMin as ReduceInstruction<P>>::null_accumulator(argmin, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceOperation::Max(max) => {
                let elements = <Max as ReduceInstruction<P>>::null_accumulator(max, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let elements = <Min as ReduceInstruction<P>>::null_accumulator(min, line_size);

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
        }
    }

    fn read_accumulator(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
    ) -> (Line<P::EI>, ReduceCoordinate) {
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
            ReduceOperation::Max(max) => {
                <Max as ReduceInstruction<P>>::read_accumulator(max, &accumulator.elements)
            }
            ReduceOperation::Min(min) => {
                <Min as ReduceInstruction<P>>::read_accumulator(min, &accumulator.elements)
            }
        }
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        destination.elements = source.elements;
        let args = &mut destination.args;
        match args {
            CubeOption::Some(val) => *val = source.args.unwrap(),
            CubeOption::None => {}
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<P::EI>,
        coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        match this {
            ReduceOperation::Sum(sum) => {
                let elements = <Sum as ReduceInstruction<P>>::reduce(
                    sum,
                    &accumulator.elements,
                    item,
                    coordinate,
                    use_planes,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Prod(sum) => {
                let elements = <Prod as ReduceInstruction<P>>::reduce(
                    sum,
                    &accumulator.elements,
                    item,
                    coordinate,
                    use_planes,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Mean(sum) => {
                let elements = <Mean as ReduceInstruction<P>>::reduce(
                    sum,
                    &accumulator.elements,
                    item,
                    coordinate,
                    use_planes,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let elements = <MaxAbs as ReduceInstruction<P>>::reduce(
                    maxabs,
                    &accumulator.elements,
                    item,
                    coordinate,
                    use_planes,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let (elements, args) = <ArgMax as ReduceInstruction<P>>::reduce(
                    argmax,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    use_planes,
                );

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let (elements, args) = <ArgMin as ReduceInstruction<P>>::reduce(
                    argmin,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    use_planes,
                );

                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceOperation::Max(max) => {
                let elements = <Max as ReduceInstruction<P>>::reduce(
                    max,
                    &accumulator.elements,
                    item,
                    coordinate,
                    use_planes,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let elements = <Min as ReduceInstruction<P>>::reduce(
                    min,
                    &accumulator.elements,
                    item,
                    coordinate,
                    use_planes,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
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
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Prod(prod) => {
                let elements = <Prod as ReduceInstruction<P>>::fuse_accumulators(
                    prod,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Mean(mean) => {
                let elements = <Mean as ReduceInstruction<P>>::fuse_accumulators(
                    mean,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::MaxAbs(maxabs) => {
                let elements = <MaxAbs as ReduceInstruction<P>>::fuse_accumulators(
                    maxabs,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::ArgMax(argmax) => {
                let (elements, args) = <ArgMax as ReduceInstruction<P>>::fuse_accumulators(
                    argmax,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceOperation::ArgMin(argmin) => {
                let (elements, args) = <ArgMin as ReduceInstruction<P>>::fuse_accumulators(
                    argmin,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceOperation::Max(max) => {
                let elements = <Max as ReduceInstruction<P>>::fuse_accumulators(
                    max,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceOperation::Min(min) => {
                let elements = <Min as ReduceInstruction<P>>::fuse_accumulators(
                    min,
                    lhs.elements,
                    rhs.elements,
                );
                DynamicAccumulatorItem::<P::EA> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
        }
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Dynamic can be implemented by fusing a Sum reduction and a element-wise division.
    fn merge_line<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Out {
        match this {
            ReduceOperation::Sum(sum) => <Sum as ReduceInstruction<P>>::merge_line::<Out>(
                sum,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::Prod(prod) => <Prod as ReduceInstruction<P>>::merge_line::<Out>(
                prod,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::Mean(mean) => <Mean as ReduceInstruction<P>>::merge_line::<Out>(
                mean,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::MaxAbs(maxabs) => <MaxAbs as ReduceInstruction<P>>::merge_line::<Out>(
                maxabs,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::ArgMax(argmax) => <ArgMax as ReduceInstruction<P>>::merge_line::<Out>(
                argmax,
                (accumulator.elements, accumulator.args.unwrap()),
                shape_axis_reduce,
            ),
            ReduceOperation::ArgMin(argmin) => <ArgMin as ReduceInstruction<P>>::merge_line::<Out>(
                argmin,
                (accumulator.elements, accumulator.args.unwrap()),
                shape_axis_reduce,
            ),
            ReduceOperation::Max(max) => <Max as ReduceInstruction<P>>::merge_line::<Out>(
                max,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceOperation::Min(min) => <Min as ReduceInstruction<P>>::merge_line::<Out>(
                min,
                accumulator.elements,
                shape_axis_reduce,
            ),
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
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
