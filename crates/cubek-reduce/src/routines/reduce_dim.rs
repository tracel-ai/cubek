use crate::{ReducePrecision, components::args::NumericLine, routines::GlobalReduceBlueprint};
use cubecl::{
    prelude::{ReadWrite, *},
    std::tensor::r#virtual::VirtualTensor,
};

#[cube]
pub trait ReduceDimRoutine {
    type Config;

    fn execute<P: ReducePrecision, Out: NumericLine>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        axis_reduce: u32,
        reduce_index: u32,
        #[comptime] config: Self::Config,
    );

    fn create_config(#[comptime] blueprint: GlobalReduceBlueprint) -> comptime_type!(Self::Config);
}
