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
    reduce_kernel_inner::<(In, Acc), Out, ReduceOperation>(
        input,
        output,
        axis_reduce,
        blueprint,
        config,
    )
}

#[cube]
fn reduce_kernel_inner<P: ReducePrecision, Out: Numeric, R: ReduceFamily>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
) {
    let inst = &R::Instruction::<P>::from_config(config);

    match comptime!(blueprint.global) {
        GlobalReduceBlueprint::Cube(cube) => {
            GlobalFullCubeReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                inst,
                blueprint.line_mode,
                cube,
            )
        }
        GlobalReduceBlueprint::FullPlane(plane) => {
            GlobalFullPlaneReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                inst,
                blueprint.line_mode,
                plane,
            )
        }
        GlobalReduceBlueprint::FullUnit(unit) => {
            GlobalFullUnitReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                inst,
                blueprint.line_mode,
                unit,
            )
        }
    };
}
