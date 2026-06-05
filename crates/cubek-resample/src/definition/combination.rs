use cubecl::prelude::*;

#[cube]
pub trait CombinationOp: Send + Sync + 'static {
    fn combine<C: Numeric>(x: C, w: C) -> C;
}

#[derive(Clone, Copy)]
pub struct IdentityCombine;

#[cube]
impl CombinationOp for IdentityCombine {
    fn combine<C: Numeric>(x: C, _w: C) -> C {
        x
    }
}
