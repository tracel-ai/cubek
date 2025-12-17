use crate::{
    backward_weight::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    components::{ConvGemmConfig as _, global::args::RuntimeArgsLaunch},
};
use cubecl::prelude::TensorHandleRef;
use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::{
    definition::{MatmulElems, MatmulLineSizes, TilingBlueprint},
    launch::{
        InputArg, InputRuntimeArg, MatmulArgs, MatmulInputHandleRef, OutputArg, OutputRuntimeArg,
    },
};

use crate::{
    components::{ConvSetupError, ConvolutionProblem, global::entry_point::ConvolutionLaunch},
    kernels::forward::algorithm::Algorithm,
};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    out_grad: &MatmulInputHandleRef<'_, R>,
    weight_grad: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    selection: TilingBlueprint,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError>
where
    InputArg<A::Args>: ConcreteInputsFactory,
    OutputArg<A::Args>: ConcreteOutputFactory,
{
    let config = A::expand_config(client, &problem, &selection, &line_sizes, dtypes)?;

    let (input, runtime_args) = <InputArg<A::Args> as ConcreteInputsFactory>::create(
        client,
        input,
        out_grad,
        &selection,
        &problem,
        &line_sizes,
        config,
        dtypes,
    );
    let output = <OutputArg<A::Args> as ConcreteOutputFactory>::create(
        client,
        weight_grad,
        &selection,
        &problem,
        &line_sizes,
        config,
    );

    let result = unsafe {
        A::GlobalConvolution::launch_unchecked::<A::Args, R>(
            client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            input,
            output,
            runtime_args,
            config,
            dtypes,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(ConvSetupError::Launch(err)),
    }
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel_virtual<'a, MA: MatmulArgs, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    runtime_args: RuntimeArgsLaunch<'a, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    selection: TilingBlueprint,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let config = A::expand_config(client, &problem, &selection, &line_sizes, dtypes)?;

    let result = unsafe {
        A::GlobalConvolution::launch_unchecked::<MA, R>(
            client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            input,
            output,
            runtime_args,
            config,
            dtypes,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(ConvSetupError::Launch(err)),
    }
}
