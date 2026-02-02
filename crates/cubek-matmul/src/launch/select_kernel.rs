use crate::definition::MatmulLineSizes;
use crate::definition::MatmulProblem;
use crate::definition::MatmulSetupError;
use crate::launch::handle::MatmulInputHandleRef;
use crate::launch::{
    ConcreteInputsFactory, ConcreteOutputFactory, InputArg, InputRuntimeArg, MatmulArgs, OutputArg,
    OutputRuntimeArg,
};
use crate::routines::LaunchInfo;
use crate::routines::{BlueprintStrategy, Routine};
use crate::{definition::MatmulElems, launch::ConfigRuntimeArg};
use cubecl::prelude::TensorHandleRef;
use cubecl::{Runtime, client::ComputeClient};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MA: MatmulArgs<Config = ()>, R: Runtime, A: Routine<()>>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory<A>,
    OutputArg<MA>: ConcreteOutputFactory<A>,
{
    let mut view_line_sizes = line_sizes;

    if let MatmulInputHandleRef::Quantized { scheme, .. } = lhs {
        view_line_sizes.lhs *= scheme.num_quants();
    }
    if let MatmulInputHandleRef::Quantized { scheme, .. } = rhs {
        view_line_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_line_sizes);
    let expand_info = A::expand_blueprint(&problem, &device_settings, blueprint_strategy)?;
    let launch_info = A::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<MA> as ConcreteInputsFactory<A>>::create(
        client,
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );
    let output = <OutputArg<MA> as ConcreteOutputFactory<A>>::create(
        client,
        out,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );

    launch_kernel::<MA, R, A>(client, input, output, (), launch_info)
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel_virtual<'a, MA: MatmulArgs, R: Runtime, A: Routine<MA::Config>>(
    client: &ComputeClient<R>,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    config: ConfigRuntimeArg<'a, MA, R>,
    problem: MatmulProblem,
    view_line_sizes: MatmulLineSizes,
    blueprint_strategy: &BlueprintStrategy<MA::Config, A>,
) -> Result<(), MatmulSetupError> {
    let device_settings = A::device_settings(client, view_line_sizes);
    let expand_info = A::expand_blueprint(&problem, &device_settings, blueprint_strategy)?;
    let launch_info = A::prepare(&problem, &device_settings, expand_info)?;

    launch_kernel::<MA, R, A>(client, input, output, config, launch_info)
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel<'a, MA: MatmulArgs, R: Runtime, A: Routine<MA::Config>>(
    client: &ComputeClient<R>,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    config: ConfigRuntimeArg<'a, MA, R>,
    launch_info: LaunchInfo<A::Blueprint>,
) -> Result<(), MatmulSetupError> {
    A::launch::<MA, R>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        input,
        output,
        config,
        launch_info.cube_count_plan.as_args(),
        launch_info.blueprint,
        &launch_info.dtypes,
    )
}
