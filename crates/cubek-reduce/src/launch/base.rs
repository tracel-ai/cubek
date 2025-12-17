use crate::{
    LineMode, ReduceError, ReducePrecision,
    components::{
        args::{ReduceArgs, TensorArgs, init_tensors},
        global::{
            cube::GlobalFullCubeReduce, plane::GlobalFullPlaneReduce, unit::GlobalFullUnitReduce,
        },
        instructions::*,
    },
    launch::{ReduceStrategy, RoutineStrategy, generate_line_size},
    routines::{
        GlobalReduceBlueprint, ReduceBlueprint, ReduceLineSettings, ReduceProblem, Routine,
        cube::CubeRoutine, plane::PlaneRoutine, unit::UnitRoutine,
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(Clone, Copy, Debug)]
pub struct ReduceDtypes {
    pub input: StorageType,
    pub output: StorageType,
    pub accumulation: StorageType,
}

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
) -> Result<(), ReduceError> {
    let problem = ReduceProblem {
        vector_size: input.shape[axis as usize] as u32,
        vector_count: output.shape.iter().map(|i| *i as u32).product(),
        axis,
        dtypes,
    };
    let line_mode = match input.strides[axis as usize] {
        1 => LineMode::Parallel,
        _ => LineMode::Perpendicular,
    };
    let (line_size_input, line_size_output) = generate_line_size::<Run>(
        client,
        &input,
        &output,
        axis as usize,
        problem.dtypes.input,
        line_mode,
        &strategy.line_size,
    );
    let settings = ReduceLineSettings {
        line_mode,
        line_size_input,
        line_size_output,
    };

    let (blueprint, settings) = match strategy.routine {
        RoutineStrategy::Unit(strategy) => {
            let routine = UnitRoutine;
            routine.prepare(client, problem, settings, strategy)?
        }
        RoutineStrategy::Plane(strategy) => {
            let routine = PlaneRoutine;
            routine.prepare(client, problem, settings, strategy)?
        }
        RoutineStrategy::Cube(strategy) => {
            let routine = CubeRoutine;
            routine.prepare(client, problem, settings, strategy)?
        }
    };

    unsafe {
        reduce_kernel::launch_unchecked::<TensorArgs, Run>(
            client,
            settings.cube_count,
            settings.cube_dim,
            input.as_tensor_arg(settings.line.line_size_input),
            output.as_tensor_arg(settings.line.line_size_output),
            ScalarArg::new(axis),
            blueprint,
            inst,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
        .map_err(ReduceError::Launch)
    }
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, Acc: Numeric, RA: ReduceArgs>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<RA, In, Out>(input, output);
    reduce_kernel_virtual::<In, Out, Acc>(&input, &mut output, axis_reduce, blueprint, config);
}

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
        GlobalReduceBlueprint::Plane(plane) => {
            GlobalFullPlaneReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                axis_reduce,
                inst,
                blueprint.line_mode,
                plane,
            )
        }
        GlobalReduceBlueprint::Unit(unit) => {
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
