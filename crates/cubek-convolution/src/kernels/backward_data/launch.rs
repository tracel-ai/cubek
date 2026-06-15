use crate::{
    backward_data::args::ConcreteArgs,
    components::{ConvolutionOperation, global::args::RuntimeArgs},
    launch::ConvolutionArgs,
};
use crate::{components::ConvSetupError, kernels::backward_data::selector::launch_kernel_concrete};
use crate::{
    components::{ConvolutionProblem, Dimensionality},
    routines::Routine,
};
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_matmul::{
    definition::{AvailableVectorSizes, MatmulElems, MatmulSetupError},
    routines::BlueprintStrategy,
};
use cubek_std::{InputBinding, MatrixLayout};

/// Backward-data dispatch helper.
///
/// Called by `cubek_convolution::launch_ref` after the routine and
/// blueprint-strategy have been resolved. Backward-data does not currently
/// support the TMA reading strategy: requesting it here returns a setup error.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub(crate) fn launch_internal<R: Runtime, const N_SPATIAL: usize, Rt: Routine>(
    client: &ComputeClient<R>,
    out_grad: InputBinding<R>,
    weights: InputBinding<R>,
    in_grad: TensorBinding<R>,
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
        out_grad,
        weights,
        in_grad,
        (&stride, &padding, &dilation),
        dimensionality,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_with_routine<R: Runtime, Rt: Routine>(
    client: &ComputeClient<R>,
    out_grad: InputBinding<R>,
    weights: InputBinding<R>,
    in_grad: TensorBinding<R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Rt::MatmulRoutine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Rt::Args: ConcreteArgs<Rt::MatmulRoutine>,
{
    let rank = in_grad.shape.len();
    let dim_c = rank - 1;

    let n = in_grad.shape[0];
    let c = in_grad.shape[dim_c];

    let out_c = out_grad.shape()[dim_c];

    let in_shape = &in_grad.shape[1..dim_c];
    let kernel_shape = &weights.shape()[1..dim_c];
    let out_shape = &out_grad.shape()[1..dim_c];

    let op = ConvolutionOperation::BackwardData;

    let out_grad_tmp = out_grad.clone();
    let weights_tmp = weights.clone();

    let out_grad_data =
        Rt::correct_layout(client, out_grad_tmp.into_data(), dtypes.lhs_global, op)?;
    let weights_data = Rt::correct_layout(client, weights_tmp.into_data(), dtypes.rhs_global, op)?;

    let mut out_grad = out_grad.clone();
    let mut weights = weights.clone();

    *out_grad.data_mut() = out_grad_data;
    *weights.data_mut() = weights_data;

    let address_type = out_grad
        .required_address_type()
        .max(weights.required_address_type())
        .max(in_grad.required_address_type(dtypes.acc_global.size()));

    let problem = ConvolutionProblem {
        m: n * in_shape.iter().product::<usize>(),
        n: c,
        k: out_c * kernel_shape.iter().product::<usize>(),

        lhs_strides: out_grad.data().strides.clone(),
        rhs_strides: weights.data().strides.clone(),
        lhs_layout: MatrixLayout::RowMajor,
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

        padded_channels: out_c,
        operation: op,

        dimensionality,
        global_dtypes: dtypes.as_global_elems(),
        address_type,
    };

    launch_kernel::<R, Rt>(
        client,
        out_grad,
        weights,
        in_grad,
        problem,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Rt: Routine>(
    client: &ComputeClient<R>,
    out_grad: InputBinding<R>,
    weights: InputBinding<R>,
    in_grad: TensorBinding<R>,
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
        out_grad.data_elem_size(),
        weights.data_elem_size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(
        &out_grad.data().strides,
        &out_grad.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        &weights.data().strides,
        &weights.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(&in_grad.strides, &in_grad.shape);

    let vector_sizes = Rt::filter_vector_sizes(vector_sizes).pick_max()?;

    launch_kernel_concrete::<R, Rt::Args, Rt::MatmulRoutine>(
        client,
        out_grad,
        weights,
        in_grad,
        problem,
        vector_sizes,
        blueprint_strategy,
        &dtypes,
    )
}

/// Returned by the unified `launch_ref` when the requested routine is not
/// supported for backward-data. Currently only the TMA reading strategy is
/// rejected.
#[allow(dead_code)]
pub(crate) fn unsupported_tma_error() -> ConvSetupError {
    ConvSetupError::Matmul(MatmulSetupError::InvalidConfig(Box::new(
        "Data backprop doesn't yet work with current TMA tiling strategy",
    )))
}
