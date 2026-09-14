use super::{ArgAccumulator, Extremum, ReduceFamily, ValueOrder, forward_reduce_instruction};
use crate::components::{
    instructions::{ReduceOutputMode, ReduceWithIndicesFamily},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Return the minimum item, its coordinate, or both, per [`ReduceOutputMode`].
/// NaNs take precedence over non-NaN values. When indices are returned, ties
/// and multiple NaNs select the lowest coordinate.
#[derive(Debug, CubeType, Clone)]
pub struct Min {
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for Min {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

impl ReduceWithIndicesFamily for Min {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

forward_reduce_instruction! {
    Min with indices,
    config: ReduceOutputMode,
    shared_accumulator: ArgAccumulator<P>,
    from_config: |config| Min { output: config },
    core: fn extremum(this: &Self) -> Extremum {
        Extremum {
            order: ValueOrder::Ascending,
            output: this.output,
        }
    },
}
