use crate::{
    components::global::args::RuntimeArgs,
    forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
};
use cubecl::prelude::TensorHandleRef;
use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::{
    definition::{MatmulElems, MatmulLineSizes},
    routines::Routine,
};
use cubek_matmul::{
    launch::{InputArg, MatmulInputHandleRef, OutputArg},
    routines::BlueprintStrategy,
};

use crate::components::{ConvSetupError, ConvolutionProblem};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<R: Runtime, Args: ConcreteArgs<A>, A: Routine<RuntimeArgs>>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<MatmulInputHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let mut view_line_sizes = line_sizes;

    if let MatmulInputHandleRef::Quantized { scheme, .. } = input {
        view_line_sizes.lhs *= scheme.num_quants();
    }
    if let MatmulInputHandleRef::Quantized { scheme, .. } = weight {
        view_line_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_line_sizes);
    let expand_info = A::expand_blueprint(
        &problem.as_matmul_problem(),
        &device_settings,
        blueprint_strategy,
    )?;

    let problem = Args::adjust_problem(client, problem, &expand_info.blueprint, dtypes);
    let launch_info = A::prepare(&problem.as_matmul_problem(), &device_settings, expand_info)?;

    let (input, runtime_args) = <InputArg<Args> as ConcreteInputsFactory<A>>::create(
        client,
        input,
        weight,
        bias.as_ref(),
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );
    let output = <OutputArg<Args> as ConcreteOutputFactory<A>>::create(
        client,
        out,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );

    cubek_matmul::launch::launch_kernel::<Args, R, A>(
        client,
        input,
        output,
        runtime_args,
        launch_info,
    )
    .map_err(ConvSetupError::Matmul)
}
