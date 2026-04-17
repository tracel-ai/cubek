use std::marker::PhantomData;

use cubecl::prelude::CubeType;
use cubecl::{
    cube,
    prelude::{Array, CubePrimitive, Vector},
};

use crate::components::instructions::AccumulatorKind;
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{
        ReduceCoordinate, ReduceRequirements, ReduceStep, SharedAccumulator,
    },
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

#[derive(CubeType)]
pub struct TopkAccumulator<E: CubePrimitive> {
    pub elements: Array<E>,
    pub coordinates: Array<u32>,
}

#[derive(CubeType)]
/// Only to respect the type system. Shared Accumulator behaviour is not supported
pub struct DummyTopkSharedAccumulator<A: CubeType + Send + Sync + 'static> {
    #[cube(comptime)]
    _phantom: PhantomData<A>,
}

#[cube]
impl<A: CubeType + Send + Sync + 'static> SharedAccumulator for DummyTopkSharedAccumulator<A> {
    type Item = A;

    fn allocate(#[comptime] _length: usize, #[comptime] _coordinate: bool) -> Self {
        unreachable!()
    }

    fn read(_accumulator: &Self, _index: usize) -> Self::Item {
        unreachable!()
    }

    fn write(_accumulator: &mut Self, _index: usize, _item: Self::Item) {
        unreachable!()
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ArgTopK {
    type Accumulator = TopkAccumulator<Vector<P::EA, P::SI>>;
    type SharedAccumulator = DummyTopkSharedAccumulator<Self::Accumulator>;
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

    fn null_accumulator(_this: &Self) -> Self::Accumulator {
        todo!("argtopk")
    }

    fn assign_accumulator(
        _this: &Self,
        _destination: &mut Self::Accumulator,
        _source: &Self::Accumulator,
    ) {
        todo!("argtopk")
    }

    fn split_accumulator(
        _this: &Self,
        _accumulator: &Self::Accumulator,
    ) -> (
        AccumulatorKind<Vector<P::EI, P::SI>>,
        ReduceCoordinate<P::SI>,
    ) {
        todo!("argtopk")
    }

    fn reduce(
        _this: &Self,
        _accumulator: &Self::Accumulator,
        _item: Vector<P::EI, P::SI>,
        _coordinate: ReduceCoordinate<P::SI>,
        #[comptime] _reduce_step: ReduceStep,
    ) -> Self::Accumulator {
        todo!("reduce Not implemented")
    }

    fn fuse_accumulators(
        _this: &Self,
        _lhs: Self::Accumulator,
        _rhs: Self::Accumulator,
    ) -> Self::Accumulator {
        todo!("fuse_accumulator Not implemented")
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        _accumulator: Self::Accumulator,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        todo!("merge_vector Not implemented")
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        _accumulator: Self::Accumulator,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        todo!("to_output_perpendicular Not implemented")
    }
}
