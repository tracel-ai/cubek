use crate::components::{ConvolutionProblem, Dimensionality};
use crate::kernels::backward_weight::selector::launch_kernel_concrete;
use crate::launch::ConvolutionArgs;
use crate::{backward_weight::args::ConcreteArgs, components::ConvSetupError};
use crate::{
    components::{ConvolutionOperation, global::args::RuntimeArgs},
    routines::Routine,
};
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_matmul::{
    definition::{AvailableVectorSizes, MatmulElems},
    routines::BlueprintStrategy,
};
use cubek_std::{InputBinding, MatrixLayout};

/// Backward-weight dispatch helper.
///
/// Called by `cubek_convolution::launch_ref` after the routine and
/// blueprint-strategy have been resolved.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub(crate) fn launch_internal<R: Runtime, const N_SPATIAL: usize, Rt: Routine>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Rt::MatmulRoutine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Rt::Args: ConcreteArgs<Rt::MatmulRoutine>,
{
    let ConvolutionArgs {
        stride,
        padding,
        dilation,
    } = args;

    let dimensionality = match N_SPATIAL {
        1 => Dimensionality::Dim1,
        2 => Dimensionality::Dim2,
        3 => Dimensionality::Dim3,
        other => unimplemented!("Unsupported dimensionality {other}"),
    };

    launch_with_routine::<R, Rt>(
        client,
        input,
        out_grad,
        weight_grad,
        (&stride, &padding, &dilation),
        dimensionality,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_with_routine<R: Runtime, Rt: Routine>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Rt::MatmulRoutine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Rt::Args: ConcreteArgs<Rt::MatmulRoutine>,
{
    let rank = input.data().shape.len();
    let dim_c = rank - 1;

    let n = input.shape()[0];
    let c = input.shape()[dim_c];

    let out_c = out_grad.shape()[dim_c];

    let in_shape = &input.shape()[1..dim_c];
    let kernel_shape = &weight_grad.shape[1..dim_c];
    let out_shape = &out_grad.shape()[1..dim_c];

    let op = ConvolutionOperation::BackwardWeight;

    let input_data = Rt::correct_layout(client, input.clone().into_data(), dtypes.lhs_global, op)?;
    let out_grad_data =
        Rt::correct_layout(client, out_grad.clone().into_data(), dtypes.rhs_global, op)?;

    let mut input = input.clone();
    let mut out_grad = out_grad.clone();

    *input.data_mut() = input_data;
    *out_grad.data_mut() = out_grad_data;

    let address_type = input
        .required_address_type()
        .max(out_grad.required_address_type())
        .max(weight_grad.required_address_type(dtypes.acc_global.size()));

    let problem = ConvolutionProblem {
        m: out_c,
        n: c * kernel_shape.iter().product::<usize>(),
        k: n * out_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.clone(),
        rhs_strides: out_grad.data().strides.clone(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: kernel_shape.iter().map(|it| *it as u32).collect(),
        stride: stride.iter().map(|it| *it as u32).collect(),
        padding: padding.iter().map(|it| *it as i32).collect(),
        dilation: dilation.iter().map(|it| *it as u32).collect(),

        batches: n,
        in_shape: in_shape.into(),
        out_shape: out_shape.into(),
        channels: c,
        out_channels: out_c,

        padded_channels: c,
        operation: op,

        dimensionality,
        global_dtypes: dtypes.as_global_elems(),
        address_type,
    };

    launch_kernel::<R, Rt>(
        client,
        input,
        out_grad,
        weight_grad,
        problem,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Rt: Routine>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    problem: ConvolutionProblem,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Rt::MatmulRoutine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Rt::Args: ConcreteArgs<Rt::MatmulRoutine>,
{
    // Shape/strides are treated as k-major, with the last dim always being the contiguous one.
    // So for the sake of selecting a vector size, the shape/strides are always row-major.
    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        input.data_elem_size(),
        out_grad.data_elem_size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(
        &out_grad.data().strides,
        &out_grad.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        &input.data().strides,
        &input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(&weight_grad.strides, &weight_grad.shape);

    let vector_sizes = Rt::filter_vector_sizes(vector_sizes).pick_max()?;

    launch_kernel_concrete::<R, Rt::Args, Rt::MatmulRoutine>(
        client,
        input,
        out_grad,
        weight_grad,
        problem,
        vector_sizes,
        blueprint_strategy,
        &dtypes,
    )
}
