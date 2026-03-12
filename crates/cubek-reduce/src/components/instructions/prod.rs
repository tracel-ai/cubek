use super::{ReduceCoordinate, ReduceFamily, ReduceInstruction};
use crate::{components::instructions::ReduceRequirements, components::precision::ReducePrecision};
use cubecl::prelude::*;

#[derive(Debug, CubeType, Clone)]
pub struct Prod {}

impl ReduceFamily for Prod {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Prod {
    type AccumulatorItem = Vector<P::EA, P::SI>;
    type SharedAccumulator = SharedMemory<Vector<P::EA, P::SI>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        Prod {}
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::from_int(1))
    }

    fn null_accumulator(_this: &Self) -> Self::AccumulatorItem {
        Vector::empty().fill(P::EA::from_int(1))
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        *destination = *source;
    }

    fn read_accumulator(
        _this: &Self,
        accumulator: &Vector<P::EA, P::SI>,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        (
            Vector::cast_from(*accumulator),
            ReduceCoordinate::new_NotRequired(),
        )
    }
    fn reduce(
        _this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Vector<P::EI, P::SI>,
        _coordinate: ReduceCoordinate<P::SI>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        let item = Vector::cast_from(item);
        if use_planes {
            *accumulator * plane_prod(item)
        } else {
            *accumulator * item
        }
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        lhs * rhs
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Out {
        let mut prod = P::EA::from_int(1);
        #[unroll]
        for k in 0..accumulator.size() {
            prod *= accumulator[k];
        }
        Out::cast_from(prod)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Vector<Out, P::SI> {
        Vector::cast_from(accumulator)
    }
}
