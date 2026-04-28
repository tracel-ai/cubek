use crate::components::{ConvolutionProblem, Dimensionality};
use crate::routines::Routine;
use crate::{components::ConvSetupError, kernels::forward::selector::launch_kernel_concrete};
use crate::{
    components::ConvolutionOperation, components::global::args::RuntimeArgs,
    forward::args::ConcreteArgs, launch::ConvolutionArgs,
};
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_matmul::definition::{AvailableVectorSizes, MatmulElems};
use cubek_matmul::routines::BlueprintStrategy;
use cubek_std::{InputBinding, MatrixLayout};

/// Forward-convolution dispatch helper.
///
/// Called by `cubek_convolution::launch_ref` after the routine and
/// blueprint-strategy have been resolved. Not meant for direct external use.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub(crate) fn launch_internal<R: Runtime, const N_SPATIAL: usize, Rt: Routine>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
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
        weight,
        bias,
        out,
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
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
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

    let n = input.data().shape[0];
    let c = input.data().shape[dim_c];

    let out_c = weight.data().shape[0];

    let in_shape = &input.data().shape[1..dim_c];
    let kernel_shape = &weight.data().shape[1..dim_c];
    let out_shape = &out.shape[1..dim_c];

    let op = ConvolutionOperation::Forward;

    let input_data = Rt::correct_layout(client, input.clone().into_data(), dtypes.lhs_global, op)?;
    let weight_data =
        Rt::correct_layout(client, weight.clone().into_data(), dtypes.rhs_global, op)?;

    let mut input = input.clone();
    let mut weight = weight.clone();

    *input.data_mut() = input_data;
    *weight.data_mut() = weight_data;

    let address_type = input
        .required_address_type()
        .max(weight.required_address_type())
        .max(
            bias.clone()
                .map(|bias| bias.required_address_type())
                .unwrap_or_default(),
        )
        .max(out.required_address_type(dtypes.acc_global.size()));

    let problem = ConvolutionProblem {
        m: n * out_shape.iter().product::<usize>(),
        n: out_c,
        k: c * kernel_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.clone(),
        rhs_strides: weight.data().strides.clone(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
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
        weight,
        bias,
        out,
        problem,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Rt: Routine>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
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
        weight.data_elem_size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(
        &input.data().strides,
        &input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        &weight.data().strides,
        &weight.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(&out.strides, &out.shape);

    let mut vector_sizes = Rt::filter_vector_sizes(vector_sizes).pick_max()?;

    // The large vector size resulting from dequantizing ends up slower due to restrictions on
    // algorithms. Use this as a quick and dirty fix.
    if input.scale().is_some() {
        vector_sizes.lhs = 1;
    }
    if weight.scale().is_some() {
        vector_sizes.rhs = 1;
    }

    launch_kernel_concrete::<R, Rt::Args, Rt::MatmulRoutine>(
        client,
        input,
        weight,
        bias,
        out,
        problem,
        vector_sizes,
        blueprint_strategy,
        &dtypes,
    )
}
