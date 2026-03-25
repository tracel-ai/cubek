use crate::definition::MatmulProblem;
use crate::definition::{AvailableVectorSizes, MatmulElems, TilingBlueprint};
use crate::definition::{MatmulAvailabilityError, MatmulSetupError};
use crate::launch::launch_kernel_concrete;
use crate::launch::{
    ConcreteInputsFactory, ConcreteOutputFactory, InputArg, MatmulArgs, OutputArg, TensorArgs,
    TensorMapArgs,
};
use crate::routines::{BlueprintStrategy, Routine};
use cubecl::features::TypeUsage;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::{Runtime, client::ComputeClient, frontend::TensorBinding};
use cubek_std::InputBinding;

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, A: Routine<()>>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs = if matrix_batch_layout(&lhs.data().strides, lhs.scheme())
        == MatrixBatchLayout::HighlyPermuted
    {
        lhs.into_contiguous(client)?
    } else {
        lhs
    };

    let rhs = if matrix_batch_layout(&rhs.data().strides, rhs.scheme())
        == MatrixBatchLayout::HighlyPermuted
    {
        rhs.into_contiguous(client)?
    } else {
        rhs
    };

    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        lhs.data_elem_size(),
        rhs.data_elem_size(),
        dtypes.acc_global.size(),
    );
    launch_inner_ref::<R, TensorArgs, A>(
        client,
        lhs,
        rhs,
        out,
        blueprint_strategy,
        vector_sizes,
        dtypes,
    )
}

/// Launch a matrix multiplication kernel, with TMA restrictions enabled.
/// TMA doesn't support permuted batches, so checks are slightly different.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref_tma<R: Runtime, A: Routine<(), Blueprint = TilingBlueprint>>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs = match matrix_batch_layout(&lhs.data().strides, lhs.scheme()) {
        MatrixBatchLayout::Contiguous
        | MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: false,
        } => lhs,
        MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: true,
        }
        | MatrixBatchLayout::HighlyPermuted => lhs.into_contiguous(client)?,
    };

    let rhs = match matrix_batch_layout(&rhs.data().strides, rhs.scheme()) {
        MatrixBatchLayout::Contiguous
        | MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: false,
        } => rhs,
        MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: true,
        }
        | MatrixBatchLayout::HighlyPermuted => rhs.into_contiguous(client)?,
    };

    let vector_sizes = AvailableVectorSizes::from_type_size_tma(client, dtypes.acc_global.size());
    launch_inner_ref::<R, TensorMapArgs, A>(
        client,
        lhs,
        rhs,
        out,
        blueprint_strategy,
        vector_sizes,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<R: Runtime, MA: MatmulArgs<Config = ()>, A: Routine<()>>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    vector_sizes: AvailableVectorSizes,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory<A>,
    OutputArg<MA>: ConcreteOutputFactory<A>,
{
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let problem = MatmulProblem::from_shapes_and_strides(
        lhs.shape().into(),
        rhs.shape().into(),
        out.shape.clone(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        out.strides.clone(),
        dtypes.as_global_elems(),
        address_type,
        lhs.scheme(),
        rhs.scheme(),
    )?;

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

    let mut vector_sizes = vector_sizes
        .filter_lhs_with_tensor(&problem.lhs_strides, &problem.lhs_shape, problem.lhs_layout)
        .filter_rhs_with_tensor(&problem.rhs_strides, &problem.rhs_shape, problem.rhs_layout)
        .filter_out_with_tensor(&problem.out_strides, &problem.out_shape)
        .pick_max()?;

    // The large vector size resulting from dequantizing ends up slower due to restrictions on
    // algorithms. Use this as a quick and dirty fix.
    if lhs.scale().is_some() {
        vector_sizes.lhs = 1;
    }
    if rhs.scale().is_some() {
        vector_sizes.rhs = 1;
    }

    launch_kernel_concrete::<MA, R, A>(
        client,
        lhs,
        rhs,
        out,
        problem,
        vector_sizes,
        blueprint_strategy,
        dtypes,
    )
}
