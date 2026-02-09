use super::{ReduceCoordinate, ReduceFamily, ReduceInstruction, ReduceRequirements, Sum};
use crate::components::precision::ReducePrecision;
use cubecl::prelude::*;

#[derive(Debug, CubeType, Clone)]
pub struct Mean {
    pub(crate) sum: Sum,
}

impl ReduceFamily for Mean {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
fn null_input<P: ReducePrecision, SI: ReduceInstruction<P>>(
    sum: &SI,
    #[comptime] line_size: LineSize,
) -> Line<P::EI> {
    SI::null_input(sum, line_size)
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Mean {
    type AccumulatorItem = Line<P::EA>;
    type SharedAccumulator = SharedMemory<Line<P::EA>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }
    fn from_config(_config: Self::Config) -> Self {
        Mean { sum: Sum {} }
    }

    fn null_input(this: &Self, #[comptime] line_size: LineSize) -> Line<P::EI> {
        <Sum as ReduceInstruction<P>>::null_input(&this.sum, line_size)
    }

    fn null_accumulator(this: &Self, #[comptime] line_size: LineSize) -> Self::AccumulatorItem {
        <Sum as ReduceInstruction<P>>::null_accumulator(&this.sum, line_size)
    }

    fn assign_accumulator(
        this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        <Sum as ReduceInstruction<P>>::assign_accumulator(&this.sum, destination, source);
    }

    fn read_accumulator(
        _this: &Self,
        accumulator: &Line<P::EA>,
    ) -> (Line<P::EI>, ReduceCoordinate) {
        (
            Line::cast_from(*accumulator),
            ReduceCoordinate::new_NotRequired(),
        )
    }

    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<P::EI>,
        _coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        <Sum as ReduceInstruction<P>>::reduce(&this.sum, accumulator, item, _coordinate, use_planes)
    }

    fn fuse_accumulators(
        this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        <Sum as ReduceInstruction<P>>::fuse_accumulators(&this.sum, lhs, rhs)
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Mean can be implemented by fusing a <Sum as ReduceInstruction<P>> reduction and a element-wise division.
    fn merge_line<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: LineSize,
    ) -> Out {
        let sum = <Sum as ReduceInstruction<P>>::merge_line::<P::EA>(
            &this.sum,
            accumulator,
            shape_axis_reduce,
        );

        Out::cast_from(sum / P::EA::cast_from(shape_axis_reduce))
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: LineSize,
    ) -> Line<Out> {
        let sum = <Sum as ReduceInstruction<P>>::to_output_perpendicular::<P::EA>(
            &this.sum,
            accumulator,
            shape_axis_reduce,
        );
        Line::cast_from(sum / Line::cast_from(shape_axis_reduce))
    }
}
