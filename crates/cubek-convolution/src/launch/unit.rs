use cubecl::{
    client::Client,
    prelude::{Scalar, TensorBinding},
};
use cubek_matmul::definition::{AccumulatorOperand, MatmulElems, MatmulProblem};
use cubek_std::{InputBinding, MatrixLayout, config::MatmulConfig};

use crate::{
    ConvolutionArgs, ConvolutionInputs,
    components::{ConvSetupError, ConvolutionOperation},
    kernels::{backward_data, backward_weight, forward},
    launch::UnitStrategy,
    routines::{Routine, unit::UnitConv},
};

/// Launch F32 register-tiled implicit convolution without accelerated matrix instructions.
///
/// Supports forward (including bias), data gradients and weight gradients.
#[allow(clippy::result_large_err)]
pub(crate) fn launch<const N: usize>(
    strategy: UnitStrategy,
    client: &Client,
    inputs: ConvolutionInputs,
    args: ConvolutionArgs<N>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    if !(1..=3).contains(&N)
        || args.stride.contains(&0)
        || args.dilation.contains(&0)
        || dtypes.lhs_global != f32::elem_type_native()
        || dtypes.rhs_global != f32::elem_type_native()
        || dtypes.acc_global != f32::elem_type_native()
    {
        return Err(ConvSetupError::Unknown);
    }
    let valid = match &inputs {
        ConvolutionInputs::Forward {
            input,
            weight,
            bias,
            out,
        } => {
            normal(input, N + 2)
                && normal(weight, N + 2)
                && tensor(out, N + 2)
                && input.shape()[0] == out.shape[0]
                && input.shape()[N + 1] == weight.shape()[N + 1]
                && weight.shape()[0] == out.shape[N + 1]
                && bias
                    .as_ref()
                    .is_none_or(|bias| normal(bias, 1) && bias.shape()[0] == out.shape[N + 1])
        }
        ConvolutionInputs::BackwardData {
            out_grad,
            weights,
            in_grad,
        } => {
            normal(out_grad, N + 2)
                && normal(weights, N + 2)
                && tensor(in_grad, N + 2)
                && out_grad.shape()[0] == in_grad.shape[0]
                && out_grad.shape()[N + 1] == weights.shape()[0]
                && in_grad.shape[N + 1] == weights.shape()[N + 1]
        }
        ConvolutionInputs::BackwardWeight {
            input,
            out_grad,
            weight_grad,
        } => {
            normal(input, N + 2)
                && normal(out_grad, N + 2)
                && tensor(weight_grad, N + 2)
                && input.shape()[0] == out_grad.shape()[0]
                && input.shape()[N + 1] == weight_grad.shape[N + 1]
                && out_grad.shape()[N + 1] == weight_grad.shape[0]
        }
    };
    if !valid {
        return Err(ConvSetupError::Unknown);
    }
    let align = client.properties().io_width() as usize / 32;
    // The forced blueprint must use the same channel padding as TensorArgs,
    // so its reduction-axis tail check stays valid for odd channel counts.
    let (m, n, k, lhs_layout, rhs_layout) = match &inputs {
        ConvolutionInputs::Forward {
            input, weight, out, ..
        } => (
            out.shape[..N + 1].iter().product(),
            out.shape[N + 1],
            weight.shape()[1..N + 1].iter().product::<usize>()
                * input.shape()[N + 1].next_multiple_of(align),
            MatrixLayout::RowMajor,
            MatrixLayout::ColMajor,
        ),
        ConvolutionInputs::BackwardData {
            out_grad,
            weights,
            in_grad,
        } => (
            in_grad.shape[..N + 1].iter().product(),
            in_grad.shape[N + 1],
            weights.shape()[1..N + 1].iter().product::<usize>()
                * out_grad.shape()[N + 1].next_multiple_of(align),
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
        ),
        ConvolutionInputs::BackwardWeight {
            input,
            out_grad,
            weight_grad,
        } => (
            out_grad.shape()[N + 1],
            weight_grad.shape[1..N + 1].iter().product::<usize>()
                * input.shape()[N + 1].next_multiple_of(align),
            out_grad.shape()[..N + 1].iter().product(),
            MatrixLayout::ColMajor,
            MatrixLayout::RowMajor,
        ),
    };
    if [m, n, k].into_iter().any(|n| n > u32::MAX as usize) {
        return Err(ConvSetupError::Unknown);
    }
    let mut problem = MatmulProblem::from_parameters_with_config(
        m,
        n,
        k,
        Vec::<usize>::new().into(),
        Vec::<usize>::new().into(),
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        None,
        None,
        dtypes.as_global_elems(),
        Default::default(),
        MatmulConfig { allow_tf32: true },
    );
    if matches!(&inputs, ConvolutionInputs::Forward { bias: Some(_), .. }) {
        problem.accumulator = AccumulatorOperand::Present;
    }
    let blueprint_strategy = strategy.blueprint(client, &problem, &dtypes);
    match inputs {
        ConvolutionInputs::Forward {
            input,
            weight,
            bias,
            out,
        } => {
            // BiasLayout reads a contiguous vector. Preserve sliced biases by
            // normalizing their storage before using the shared bias reader.
            let bias = bias
                .map(|bias| {
                    UnitConv::correct_layout(
                        client,
                        bias.into_data(),
                        f32::elem_type_native(),
                        ConvolutionOperation::Forward,
                    )
                    .map(|bias| InputBinding::new(bias, f32::elem_type_native()))
                })
                .transpose()?;
            forward::launch::launch_internal::<N, UnitConv>(
                client,
                input,
                weight,
                bias,
                out,
                args,
                &blueprint_strategy,
                dtypes,
            )
        }
        ConvolutionInputs::BackwardData {
            out_grad,
            weights,
            in_grad,
        } => backward_data::launch::launch_internal::<N, UnitConv>(
            client,
            out_grad,
            weights,
            in_grad,
            args,
            &blueprint_strategy,
            dtypes,
        ),
        ConvolutionInputs::BackwardWeight {
            input,
            out_grad,
            weight_grad,
        } => {
            let rows = out_grad.shape()[..N + 1].iter().product::<usize>();
            if dtypes.lhs_global == f32::elem_type_native()
                && backward_weight::unit::split_count(rows) > 1
            {
                return backward_weight::unit::launch_split::<N>(
                    client,
                    input,
                    out_grad,
                    weight_grad,
                    args,
                    strategy,
                    dtypes,
                );
            }
            backward_weight::launch::launch_internal::<N, UnitConv>(
                client,
                input,
                out_grad,
                weight_grad,
                args,
                &blueprint_strategy,
                dtypes,
            )
        }
    }
}

fn tensor(binding: &TensorBinding, rank: usize) -> bool {
    binding.shape.len() == rank
        && binding.strides.len() == rank
        && !binding.shape.contains(&0)
        && !binding.tiling.is_tiled()
        && binding
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .is_some_and(|n| n <= u32::MAX as usize)
}

fn normal(binding: &InputBinding, rank: usize) -> bool {
    matches!(binding, InputBinding::Normal(data, dtype) if *dtype == f32::elem_type_native() && tensor(data, rank))
}
