use cubecl::prelude::*;

#[cube]
pub trait ReductionOp: Send + Sync + 'static {
    fn identity<C: Numeric>() -> C;

    fn reduce<C: Numeric>(acc: C, val: C) -> C;
}

#[derive(Clone, Copy)]
pub struct LinearReduction;

#[cube]
impl ReductionOp for LinearReduction {
    fn identity<C: Numeric>() -> C {
        C::from_int(0)
    }

    fn reduce<C: Numeric>(acc: C, val: C) -> C {
        acc + val
    }
}
