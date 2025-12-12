use crate::{ReducePrecision, routines::GlobalReduceBlueprint};
use cubecl::{
    prelude::{Numeric, ReadWrite, *},
    std::tensor::r#virtual::VirtualTensor,
};

#[cube]
pub trait ReduceDimRoutine {
    type Config;

    fn execute<P: ReducePrecision, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        reduce_index: u32,
        #[comptime] config: Self::Config,
    );

    fn create_config(#[comptime] blueprint: GlobalReduceBlueprint) -> comptime_type!(Self::Config);
}
