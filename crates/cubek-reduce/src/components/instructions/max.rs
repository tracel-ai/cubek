use super::{ArgAccumulator, Extremum, ReduceFamily, ValueOrder, forward_reduce_instruction};
use crate::components::{
    instructions::{ReduceOutputMode, ReduceWithIndicesFamily},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

/// Return the maximum item, its coordinate, or both, per [`ReduceOutputMode`].
/// NaNs take precedence over non-NaN values. When indices are returned, ties
/// and multiple NaNs select the lowest coordinate.
#[derive(Debug, CubeType, Clone)]
pub struct Max {
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for Max {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

impl ReduceWithIndicesFamily for Max {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ReduceOutputMode;
}

forward_reduce_instruction! {
    Max with indices,
    config: ReduceOutputMode,
    shared_accumulator: ArgAccumulator<P>,
    from_config: |config| Max { output: config },
    core: fn extremum(this: &Self) -> Extremum {
        Extremum {
            order: ValueOrder::Descending,
            output: this.output,
        }
    },
}
