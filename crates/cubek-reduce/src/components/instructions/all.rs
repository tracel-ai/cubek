use super::{Predicate, ReduceFamily, ValueOrder, forward_reduce_instruction};
use crate::components::precision::ReducePrecision;
use cubecl::prelude::*;

/// Logical-AND reduction: returns `1` if every element along the reduced axis is
/// non-zero, and `0` otherwise.
///
/// Each element is normalized to a `0/1` flag, then combined with `min`. On
/// `{0, 1}`, AND is exactly `min`, so the vectorized `plane_min` machinery is
/// reused unchanged.
#[derive(Debug, CubeType, Clone)]
pub struct All;

impl ReduceFamily for All {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

forward_reduce_instruction! {
    All,
    config: (),
    shared_accumulator: Shared<[Vector<P::EA, P::SI>]>,
    from_config: |_config| All {},
    core: fn predicate(_this: &Self) -> Predicate {
        Predicate {
            order: ValueOrder::Ascending,
        }
    },
}
