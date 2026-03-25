use crate::{
    backward_data::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
    components::global::args::RuntimeArgs,
};
use cubecl::prelude::TensorBinding;
use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::{
    definition::{MatmulElems, MatmulVectorSizes},
    launch::{InputArg, OutputArg},
    routines::{BlueprintStrategy, Routine},
};
use cubek_std::InputBinding;

use crate::components::{ConvSetupError, ConvolutionProblem};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<R: Runtime, Args: ConcreteArgs<A>, A: Routine<RuntimeArgs>>(
    client: &ComputeClient<R>,
    out_grad: InputBinding<R>,
    weights: InputBinding<R>,
    in_grad: TensorBinding<R>,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let mut view_vector_sizes = vector_sizes;

    if let InputBinding::Quantized { scheme, .. } = out_grad {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = weights {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_vector_sizes);
    let expand_info = A::expand_blueprint(
        &problem.as_matmul_problem(),
        &device_settings,
        blueprint_strategy,
    )?;

    let problem = Args::adjust_problem(client, problem, &expand_info.blueprint, dtypes);
    let launch_info = A::prepare(&problem.as_matmul_problem(), &device_settings, expand_info)?;

    let (input, runtime_args) = <InputArg<Args> as ConcreteInputsFactory<A>>::create(
        out_grad,
        weights,
        &launch_info.blueprint,
        &problem,
        dtypes,
    );
    let output = <OutputArg<Args> as ConcreteOutputFactory<A>>::create(
        in_grad,
        &launch_info.blueprint,
        &problem,
    );

    let result = cubek_matmul::launch::launch_kernel::<Args, R, A>(
        client,
        input,
        output,
        runtime_args,
        launch_info,
    );

    result.map_err(ConvSetupError::Matmul)
}
