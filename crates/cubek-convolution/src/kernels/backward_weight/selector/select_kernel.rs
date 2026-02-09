use crate::{
    backward_weight::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
    components::global::args::RuntimeArgs,
};
use cubecl::prelude::TensorHandleRef;
use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::{
    definition::{MatmulElems, MatmulLineSizes},
    launch::{InputArg, MatmulInputHandleRef, OutputArg},
    routines::{BlueprintStrategy, Routine},
};

use crate::components::{ConvSetupError, ConvolutionProblem};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<R: Runtime, Args: ConcreteArgs<A>, A: Routine<RuntimeArgs>>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    out_grad: &MatmulInputHandleRef<'_, R>,
    weight_grad: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let mut view_line_sizes = line_sizes;

    if let MatmulInputHandleRef::Quantized { scheme, .. } = input {
        view_line_sizes.lhs *= scheme.num_quants();
    }
    if let MatmulInputHandleRef::Quantized { scheme, .. } = out_grad {
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
        out_grad,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );
    let output = <OutputArg<Args> as ConcreteOutputFactory<A>>::create(
        client,
        weight_grad,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
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
