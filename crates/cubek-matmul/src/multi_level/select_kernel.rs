use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
    multi_level::{
        BatchMatmulRoutine, LaunchInfo,
        args::{
            ConcreteInputsFactory, ConcreteOutputFactory, ConfigRuntimeArg, InputArg,
            InputRuntimeArg, MatmulArgs, OutputArg, OutputRuntimeArg,
        },
        definition::cube_mapping_launch,
    },
    routine::BlueprintStrategy,
};
use cubecl::{client::ComputeClient, prelude::TensorBinding};
use cubek_std::InputBinding;

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MA: MatmulArgs<Config = ()>, A: BatchMatmulRoutine<()>>(
    client: &ComputeClient,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    problem: MatmulProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory<A>,
    OutputArg<MA>: ConcreteOutputFactory<A>,
{
    let mut view_vector_sizes = vector_sizes;

    if let InputBinding::Quantized { scheme, .. } = lhs {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = rhs {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_vector_sizes);
    let expand_info = A::expand_blueprint(&problem, &device_settings, blueprint_strategy)?;
    let launch_info = A::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<MA> as ConcreteInputsFactory<A>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );
    let output = <OutputArg<MA> as ConcreteOutputFactory<A>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );

    launch_kernel::<MA, A>(client, input, output, (), launch_info)
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel_virtual<MA: MatmulArgs, A: BatchMatmulRoutine<MA::Config>>(
    client: &ComputeClient,
    input: InputRuntimeArg<MA>,
    output: OutputRuntimeArg<MA>,
    config: ConfigRuntimeArg<MA>,
    problem: MatmulProblem,
    view_vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<MA::Config, A>,
) -> Result<(), MatmulSetupError> {
    let device_settings = A::device_settings(client, view_vector_sizes);
    let expand_info = A::expand_blueprint(&problem, &device_settings, blueprint_strategy)?;
    let launch_info = A::prepare(&problem, &device_settings, expand_info)?;

    launch_kernel::<MA, A>(client, input, output, config, launch_info)
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel<MA: MatmulArgs, A: BatchMatmulRoutine<MA::Config>>(
    client: &ComputeClient,
    input: InputRuntimeArg<MA>,
    output: OutputRuntimeArg<MA>,
    config: ConfigRuntimeArg<MA>,
    launch_info: LaunchInfo<A::Blueprint>,
) -> Result<(), MatmulSetupError> {
    A::launch::<MA>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        launch_info.address_type,
        input,
        output,
        config,
        cube_mapping_launch(&launch_info.cube_count_plan),
        launch_info.blueprint,
        &launch_info.dtypes,
        &launch_info.vector_sizes,
    )
}
