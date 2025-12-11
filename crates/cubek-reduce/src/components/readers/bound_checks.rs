use crate::{BoundChecksInner, ReduceInstruction, ReducePrecision};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords1d},
};

#[derive(CubeType)]
#[allow(unused)]
pub enum ReaderBoundChecks<P: ReducePrecision> {
    NotRequired,
    Required(RequiredReaderBoundChecks<P>),
}

#[derive(CubeType)]
pub struct RequiredReaderBoundChecks<P: ReducePrecision> {
    #[cube(comptime)]
    bound_checks: BoundChecksInner,
    pos_max: u32,
    null_input: Line<P::EI>,
}

#[cube]
impl<P: ReducePrecision> ReaderBoundChecks<P> {
    pub fn new<I: ReduceInstruction<P>>(
        inst: &I,
        pos_max: u32,
        #[comptime] line_size: u32,
        #[comptime] bound_checks: BoundChecksInner,
    ) -> ReaderBoundChecks<P> {
        match comptime!(bound_checks) {
            BoundChecksInner::None => ReaderBoundChecks::new_NotRequired(),
            BoundChecksInner::Mask | BoundChecksInner::Branch => {
                let pos_max = pos_max;
                let null_input = I::null_input(inst, line_size);

                ReaderBoundChecks::new_Required(RequiredReaderBoundChecks::<P> {
                    bound_checks,
                    pos_max,
                    null_input,
                })
            }
        }
    }
    pub fn read(&self, pos: u32, offset: u32, view: &View<Line<P::EI>, Coords1d>) -> Line<P::EI> {
        match self {
            ReaderBoundChecks::NotRequired => view[offset],
            ReaderBoundChecks::Required(checks) => match comptime!(checks.bound_checks) {
                BoundChecksInner::None => view[offset],
                BoundChecksInner::Mask => {
                    let mask = pos < checks.pos_max;
                    let index = offset * u32::cast_from(mask);
                    select(mask, view[index], checks.null_input)
                }
                BoundChecksInner::Branch => {
                    if pos < checks.pos_max {
                        view[offset]
                    } else {
                        checks.null_input
                    }
                }
            },
        }
    }
}
