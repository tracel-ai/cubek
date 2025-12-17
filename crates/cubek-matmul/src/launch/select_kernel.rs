use crate::components::batch::BatchConfig;
use crate::definition::MatmulElems;
use crate::definition::MatmulLineSizes;
use crate::definition::MatmulProblem;
use crate::definition::MatmulSetupError;
use crate::launch::handle::MatmulInputHandleRef;
use crate::launch::{
    ConcreteInputsFactory, ConcreteOutputFactory, InputArg, InputRuntimeArg, MatmulArgs, OutputArg,
    OutputRuntimeArg,
};
use crate::routines::{BlueprintStrategy, Routine};
use cubecl::prelude::TensorHandleRef;
use cubecl::{Runtime, client::ComputeClient};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MA: MatmulArgs, R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    plane_dim: u32,
    blueprint_strategy: &BlueprintStrategy<A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory<A>,
    OutputArg<MA>: ConcreteOutputFactory<A>,
{
    let mut view_line_sizes = line_sizes;

    if let MatmulInputHandleRef::Quantized { scheme, .. } = lhs {
        view_line_sizes.lhs *= scheme.num_quants() as u8;
    }
    if let MatmulInputHandleRef::Quantized { scheme, .. } = rhs {
        view_line_sizes.rhs *= scheme.num_quants() as u8;
    }

    // Prefer output type for stage because it's the same size at best, but often smaller.
    // Having stage == global also enables things like TMA, and an f16 stage for output enables
    // using `stmatrix` on the registers after casting.
    if A::can_cast_stage_element() {
        dtypes.lhs_stage.dtype = dtypes.lhs_global.dtype;
        dtypes.rhs_stage.dtype = dtypes.rhs_global.dtype;
        dtypes.acc_stage.dtype = dtypes.acc_global.dtype;
    }

    let blueprint = match blueprint_strategy {
        BlueprintStrategy::Forced(selection) => selection.clone(),
        BlueprintStrategy::Inferred(args) => {
            A::prepare(client, &problem, plane_dim, &view_line_sizes, args, dtypes)?
        }
    };
    let config = A::expand_config(client, &problem, &blueprint, &view_line_sizes, dtypes)?;
    let cube_count_plan = config.cube_count_plan(
        &problem,
        &client.properties().hardware.max_cube_count.clone(),
    );

    A::launch::<MA, R>(
        client,
        config.cube_dim(),
        cube_count_plan.resolve(),
        <InputArg<MA> as ConcreteInputsFactory<A>>::create(
            client,
            lhs,
            rhs,
            &blueprint,
            &problem,
            &line_sizes,
            config,
            dtypes,
        ),
        <OutputArg<MA> as ConcreteOutputFactory<A>>::create(
            client,
            out,
            &blueprint,
            &problem,
            &line_sizes,
            config,
            dtypes,
        ),
        cube_count_plan.as_args(),
        config,
        dtypes,
    )
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel_virtual<'a, MA: MatmulArgs, R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    problem: MatmulProblem,
    view_line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &BlueprintStrategy<A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    // Prefer output type for stage because it's the same size at best, but often smaller.
    // Having stage == global also enables things like TMA, and an f16 stage for output enables
    // using `stmatrix` on the registers after casting.
    if A::can_cast_stage_element() {
        dtypes.lhs_stage.dtype = dtypes.lhs_global.dtype;
        dtypes.rhs_stage.dtype = dtypes.rhs_global.dtype;
        dtypes.acc_stage.dtype = dtypes.acc_global.dtype;
    }

    let selection = match selection {
        BlueprintStrategy::Forced(selection) => selection.clone(),
        BlueprintStrategy::Inferred(args) => {
            A::prepare(client, &problem, plane_dim, &view_line_sizes, args, dtypes)?
        }
    };
    let config = A::expand_config(client, &problem, &selection, &view_line_sizes, dtypes)?;

    let cube_count_plan = config.cube_count_plan(
        &problem,
        &client.properties().hardware.max_cube_count.clone(),
    );

    A::launch::<MA, R>(
        client,
        config.cube_dim(),
        cube_count_plan.resolve(),
        input,
        output,
        cube_count_plan.as_args(),
        config,
        dtypes,
    )
}
