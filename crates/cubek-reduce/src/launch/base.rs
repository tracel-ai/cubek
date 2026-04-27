use crate::{
    ReduceError, ReducePrecision, VectorizationMode,
    components::{
        args::{NumericVector, ReduceArgs, TensorArgs, init_tensors},
        global::{
            cube::GlobalFullCubeReduce, plane::GlobalFullPlaneReduce, unit::GlobalFullUnitReduce,
        },
        instructions::*,
    },
    launch::{ReduceStrategy, RoutineStrategy, generate_vector_size},
    output_vectorization_axis,
    routines::{
        GlobalReduceBlueprint, ReduceBlueprint, ReduceProblem, ReduceVectorSettings, Routine,
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
    input: TensorBinding<Run>,
    output: TensorBinding<Run>,
    reduce_axis: usize,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
) -> Result<(), ReduceError> {
    let address_type = input
        .required_address_type(dtypes.input.size())
        .max(output.required_address_type(dtypes.output.size()));

    // Number of distinct reductions = product of non-reduce input dims.
    let reduce_len = input.shape[reduce_axis];
    let input_elems: usize = input.shape.iter().copied().product();
    let reduce_count = input_elems / reduce_len;

    let problem = ReduceProblem {
        reduce_len,
        reduce_count,
        axis: reduce_axis,
        dtypes,
        address_type,
    };
    let vectorization_mode = match input.strides[reduce_axis] {
        1 => VectorizationMode::Parallel,
        _ => VectorizationMode::Perpendicular,
    };

    let out_vec_axis = output_vectorization_axis(&input.strides, reduce_axis, vectorization_mode);

    let (vector_size_input, vector_size_output) = generate_vector_size::<Run>(
        client,
        &input,
        &output,
        reduce_axis,
        problem.dtypes.input,
        vectorization_mode,
        &strategy.vectorization,
    );
    let settings = ReduceVectorSettings {
        vectorization_mode,
        vector_size_input,
        vector_size_output,
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
            settings.address_type,
            settings.vector.vector_size_input,
            settings.vector.vector_size_output,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            reduce_axis,
            out_vec_axis,
            blueprint,
            inst,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
    };

    Ok(())
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn reduce_kernel<
    In: Numeric,
    InSize: Size,
    Out: Numeric,
    OutSize: Size,
    Acc: Numeric,
    RA: ReduceArgs,
>(
    input: &RA::Input<In, InSize>,
    output: &mut RA::Output<Out, OutSize>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<RA, In, InSize, Out, OutSize>(input, output);
    reduce_kernel_virtual::<In, InSize, Out, OutSize, Acc>(
        &input,
        &mut output,
        reduce_axis,
        out_vec_axis,
        blueprint,
        config,
    );
}

#[cube]
pub fn reduce_kernel_virtual<
    In: Numeric,
    InSize: Size,
    Out: Numeric,
    OutSize: Size,
    Acc: Numeric,
>(
    input: &VirtualTensor<In, InSize>,
    output: &mut VirtualTensor<Out, OutSize, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
) {
    reduce_kernel_inner::<(In, InSize, Acc), (Out, OutSize), ReduceOperation>(
        input,
        output,
        reduce_axis,
        out_vec_axis,
        blueprint,
        config,
    )
}

#[cube]
fn reduce_kernel_inner<P: ReducePrecision, Out: NumericVector, R: ReduceFamily>(
    input: &VirtualTensor<P::EI, P::SI>,
    output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
) {
    let inst = &R::Instruction::<P>::from_config(config);

    match blueprint.global {
        GlobalReduceBlueprint::Cube(cube) => {
            GlobalFullCubeReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                inst,
                blueprint.vectorization_mode,
                cube,
            )
        }
        GlobalReduceBlueprint::Plane(plane) => {
            GlobalFullPlaneReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                inst,
                blueprint.vectorization_mode,
                plane,
            )
        }
        GlobalReduceBlueprint::Unit(unit) => {
            GlobalFullUnitReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                inst,
                blueprint.vectorization_mode,
                unit,
            )
        }
    };
}
