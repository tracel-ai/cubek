use super::{Predicate, ReduceFamily, ValueOrder, forward_reduce_instruction};
use crate::components::precision::ReducePrecision;
use cubecl::prelude::*;

/// Logical-OR reduction: returns `1` if any element along the reduced axis is
/// non-zero, and `0` otherwise.
///
/// Each element is normalized to a `0/1` flag, then combined with `max`. On
/// `{0, 1}`, OR is exactly `max`, so the vectorized `plane_max` machinery is
/// reused unchanged.
#[derive(Debug, CubeType, Clone)]
pub struct Any;

impl ReduceFamily for Any {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

forward_reduce_instruction! {
    Any,
    config: (),
    shared_accumulator: Shared<[Vector<P::EA, P::SI>]>,
    from_config: |_config| Any {},
    core: fn predicate(_this: &Self) -> Predicate {
        Predicate {
            order: ValueOrder::Descending,
        }
    },
}
