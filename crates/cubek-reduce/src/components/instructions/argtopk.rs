use std::marker::PhantomData;

use cubecl::prelude::{CubeType, Scalar, Size};
use cubecl::{
    cube,
    prelude::{Array, Vector},
};

use crate::components::instructions::{Accumulator, AccumulatorKind, Item};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{ReduceRequirements, ReduceStep, SharedAccumulator},
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
pub struct TopkAccumulator<E: Scalar, S: Size> {
    pub elements: Array<Vector<E, S>>,
    pub coordinates: Array<Vector<u32, S>>,
}

#[derive(CubeType)]
/// Only to respect the type system. Shared Accumulator behaviour is not supported
pub struct DummyTopkSharedAccumulator<A: CubeType + Send + Sync + 'static> {
    #[cube(comptime)]
    _phantom: PhantomData<A>,
}

#[cube]
impl<A: CubeType + Send + Sync + 'static, P: ReducePrecision> SharedAccumulator<P>
    for DummyTopkSharedAccumulator<A>
{
    fn allocate(#[comptime] _length: usize, #[comptime] _coordinate: bool) -> Self {
        unreachable!()
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
    type SharedAccumulator = DummyTopkSharedAccumulator<Accumulator<P>>;
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

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        todo!("argtopk")
    }

    fn assign_accumulator(
        _this: &Self,
        _destination: &mut Accumulator<P>,
        _source: &Accumulator<P>,
    ) {
        todo!("argtopk")
    }

    fn reduce(
        _this: &Self,
        _accumulator: &Accumulator<P>,
        _item: Item<P>,
        #[comptime] _reduce_step: ReduceStep,
    ) -> Accumulator<P> {
        todo!("reduce Not implemented")
    }

    fn plane_reduce_inplace(_this: &Self, _accumulator: &mut Accumulator<P>) {
        todo!()
    }

    fn fuse_accumulators(
        _this: &Self,
        _lhs: &Accumulator<P>,
        _rhs: &Accumulator<P>,
    ) -> Accumulator<P> {
        todo!("fuse_accumulator Not implemented")
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        _accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out> {
        todo!("merge_vector Not implemented")
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        _accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>> {
        todo!("to_output_perpendicular Not implemented")
    }
}
