use crate::definition::MatmulProblem;
use crate::definition::{AvailableLineSizes, MatmulElems, TilingBlueprint};
use crate::definition::{MatmulAvailabilityError, MatmulSetupError};
use crate::launch::handle::MatmulInputHandleRef;
use crate::launch::launch_kernel_concrete;
use crate::launch::{
    ConcreteInputsFactory, ConcreteOutputFactory, InputArg, MatmulArgs, OutputArg, TensorArgs,
    TensorMapArgs,
};
use crate::routines::{BlueprintStrategy, Routine};
use cubecl::features::TypeUsage;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::{Runtime, client::ComputeClient, frontend::TensorHandleRef};

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    blueprint_strategy: &BlueprintStrategy<A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs_owned;
    let lhs = if matrix_batch_layout(lhs.data().strides, lhs.scheme())
        == MatrixBatchLayout::HighlyPermuted
    {
        lhs_owned = lhs.into_contiguous(client)?;
        &lhs_owned.as_ref()
    } else {
        lhs
    };

    let rhs_owned;
    let rhs = if matrix_batch_layout(rhs.data().strides, rhs.scheme())
        == MatrixBatchLayout::HighlyPermuted
    {
        rhs_owned = rhs.into_contiguous(client)?;
        &rhs_owned.as_ref()
    } else {
        rhs
    };

    launch_inner_ref::<R, TensorArgs, A>(
        client,
        lhs,
        rhs,
        out,
        blueprint_strategy,
        AvailableLineSizes::from_type_sizes(
            client,
            lhs.data().elem_size,
            rhs.data().elem_size,
            out.elem_size,
        ),
        dtypes,
    )
}

/// Launch a matrix multiplication kernel, with TMA restrictions enabled.
/// TMA doesn't support permuted batches, so checks are slightly different.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref_tma<R: Runtime, A: Routine<Blueprint = TilingBlueprint>>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    blueprint_strategy: &BlueprintStrategy<A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs_owned;
    let lhs = match matrix_batch_layout(lhs.data().strides, lhs.scheme()) {
        MatrixBatchLayout::Contiguous
        | MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: false,
        } => lhs,
        MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: true,
        }
        | MatrixBatchLayout::HighlyPermuted => {
            lhs_owned = lhs.into_contiguous(client)?;
            &lhs_owned.as_ref()
        }
    };

    let rhs_owned;
    let rhs = match matrix_batch_layout(rhs.data().strides, rhs.scheme()) {
        MatrixBatchLayout::Contiguous
        | MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: false,
        } => rhs,
        MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: true,
        }
        | MatrixBatchLayout::HighlyPermuted => {
            rhs_owned = rhs.into_contiguous(client)?;
            &rhs_owned.as_ref()
        }
    };

    launch_inner_ref::<R, TensorMapArgs, A>(
        client,
        lhs,
        rhs,
        out,
        blueprint_strategy,
        AvailableLineSizes::from_type_size_tma(client, out.elem_size),
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<R: Runtime, MA: MatmulArgs, A: Routine>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    blueprint_strategy: &BlueprintStrategy<A>,
    line_sizes: AvailableLineSizes,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory<A>,
    OutputArg<MA>: ConcreteOutputFactory<A>,
{
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs.shape().to_vec(),
        rhs.shape().to_vec(),
        out.shape.to_vec(),
        lhs.data().strides.to_vec(),
        rhs.data().strides.to_vec(),
        out.strides.to_vec(),
        dtypes.as_global_elems(),
        lhs.scheme(),
        rhs.scheme(),
    );

    if !client
        .properties()
        .features
        .type_usage(dtypes.lhs_global)
        .contains(TypeUsage::Conversion)
        || !client
            .properties()
            .features
            .type_usage(dtypes.rhs_global)
            .contains(TypeUsage::Conversion)
        || !client
            .properties()
            .features
            .type_usage(dtypes.acc_global)
            .contains(TypeUsage::Conversion)
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable {
                lhs: dtypes.lhs_global,
                rhs: dtypes.rhs_global,
                output: dtypes.acc_global,
            },
        ));
    }

    let mut line_sizes = line_sizes
        .filter_lhs_with_tensor(&problem.lhs_strides, &problem.lhs_shape, problem.lhs_layout)
        .filter_rhs_with_tensor(&problem.rhs_strides, &problem.rhs_shape, problem.rhs_layout)
        .filter_out_with_tensor(&problem.out_strides, &problem.out_shape)
        .pick_max()?;

    // The large line size resulting from dequantizing ends up slower due to restrictions on
    // algorithms. Use this as a quick and dirty fix.
    if lhs.scale().is_some() {
        line_sizes.lhs = 1;
    }
    if rhs.scale().is_some() {
        line_sizes.rhs = 1;
    }

    launch_kernel_concrete::<MA, R, A>(
        client,
        lhs,
        rhs,
        out,
        problem,
        line_sizes,
        blueprint_strategy,
        dtypes,
    )
}
