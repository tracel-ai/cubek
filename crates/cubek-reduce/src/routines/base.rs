use crate::{
    components::{
        global::{
            cube::GlobalFullCubeReduce, plane::GlobalFullPlaneReduce, unit::GlobalFullUnitReduce,
        },
        instructions::*,
        precision::ReducePrecision,
    },
    routines::{GlobalReduceBlueprint, ReduceBlueprint},
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn reduce_kernel_virtual<In: Numeric, Out: Numeric, Acc: Numeric>(
    input: &VirtualTensor<In>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
) {
    let reduce_index = get_reduce_index(blueprint.global);

    reduce_kernel_inner::<(In, Acc), Out, ReduceOperation>(
        input,
        output,
        axis_reduce,
        reduce_index,
        blueprint,
        config,
    )
}

#[cube]
fn reduce_kernel_inner<P: ReducePrecision, Out: Numeric, R: ReduceFamily>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    reduce_index: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
) {
    let inst = &R::Instruction::<P>::from_config(config);

    match comptime!(blueprint.global) {
        GlobalReduceBlueprint::Cube(..) => {
            GlobalFullCubeReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                reduce_index,
                inst,
                blueprint,
            )
        }
        GlobalReduceBlueprint::FullPlane(..) => {
            GlobalFullPlaneReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                reduce_index,
                inst,
                blueprint,
            )
        }
        GlobalReduceBlueprint::FullUnit => {
            GlobalFullUnitReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                reduce_index,
                inst,
                blueprint,
            )
        }
    };
}

#[cube]
fn get_reduce_index(#[comptime] params: GlobalReduceBlueprint) -> u32 {
    match params {
        GlobalReduceBlueprint::FullUnit => ABSOLUTE_POS,
        GlobalReduceBlueprint::FullPlane { .. } => CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y,
        GlobalReduceBlueprint::Cube { .. } => CUBE_POS,
    }
}
