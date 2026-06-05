use super::{CombinationOp, IndexMapper, ReductionOp};
use cubecl::prelude::*;

#[cube]
pub trait GlobalOp: Send + Sync + 'static {
    type F: IndexMapper;
    type H: IndexMapper;
    type K: IndexMapper;
    type Combine: CombinationOp;
    type Reduce: ReductionOp;
}
