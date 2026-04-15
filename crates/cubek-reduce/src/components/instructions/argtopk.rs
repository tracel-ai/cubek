use cubecl::{CubeType, cube, prelude::Vector};

use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{ArgAccumulator, ReduceCoordinate, ReduceRequirements, ReduceStep},
};
use cubecl::frontend::Numeric;

#[derive(Debug, CubeType, Clone)]
pub struct ArgTopK {
    pub k: u32,
}

impl ReduceFamily for ArgTopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = u32;
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ArgTopK {
    type AccumulatorItem = (Vector<P::EA, P::SI>, Vector<u32, P::SI>);
    type SharedAccumulator = ArgAccumulator<P::EA, P::SI>;
    type Config = u32;

    fn requirements(_this: &Self) -> super::ReduceRequirements {
        ReduceRequirements { coordinates: true }
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        ArgTopK { k: config }
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        todo!("argtopk")
    }

    fn null_accumulator(_this: &Self) -> Self::AccumulatorItem {
        todo!("argtopk")
    }

    fn assign_accumulator(
        _this: &Self,
        _destination: &mut Self::AccumulatorItem,
        _source: &Self::AccumulatorItem,
    ) {
        todo!("argtopk")
    }

    fn read_accumulator(
        _this: &Self,
        _accumulator: &Self::AccumulatorItem,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        todo!("argtopk")
    }

    fn reduce(
        _this: &Self,
        _accumulator: &Self::AccumulatorItem,
        _item: Vector<P::EI, P::SI>,
        _coordinate: ReduceCoordinate<P::SI>,
        #[comptime] _reduce_step: ReduceStep,
    ) -> Self::AccumulatorItem {
        todo!("reduce Not implemented")
    }

    fn fuse_accumulators(
        _this: &Self,
        _lhs: Self::AccumulatorItem,
        _rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        todo!("fuse_accumulator Not implemented")
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        _accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Out {
        todo!("merge_vector Not implemented")
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        _accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Vector<Out, P::SI> {
        todo!("to_output_perpendicular Not implemented")
    }
}
