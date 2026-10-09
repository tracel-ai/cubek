use crate::{
    definition::{
        AvailableVectorSizes, MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError,
    },
    multi_level::{
        BatchMatmulRoutine,
        args::{
            ConcreteInputsFactory, ConcreteOutputFactory, InputArg, MatmulArgs, OutputArg,
            TensorArgs, TensorMapArgs, TensorMapStoreArgs,
        },
        definition::BatchMatmulBlueprint,
        launch_kernel_concrete,
    },
    routine::{BlueprintStrategy, into_contiguous_if_highly_permuted},
};
use cubecl::{
    features::{Tma, TypeUsage},
    std::tensor::{MatrixBatchLayout, matrix_batch_layout},
    {client::Client, frontend::TensorBinding},
};
use cubek_std::InputBinding;

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref<A: BatchMatmulRoutine<()>>(
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let lhs = into_contiguous_if_highly_permuted(client, lhs)?;
    let rhs = into_contiguous_if_highly_permuted(client, rhs)?;

    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        lhs.data_elem_size(),
        rhs.data_elem_size(),
        dtypes.acc_global.size(),
    );
    launch_inner_ref::<TensorArgs, A>(
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
pub fn launch_ref_tma<A: BatchMatmulRoutine<(), Blueprint = BatchMatmulBlueprint>>(
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let (lhs, rhs) = tma_inputs(client, lhs, rhs)?;
    let vector_sizes = AvailableVectorSizes::from_type_size_tma(client, dtypes.acc_global.size());
    launch_inner_ref::<TensorMapArgs, A>(
        client,
        lhs,
        rhs,
        out,
        blueprint_strategy,
        vector_sizes,
        dtypes,
    )
}

/// [`launch_ref_tma`], with the output written by TMA stores as well.
#[allow(clippy::result_large_err)]
pub fn launch_ref_tma_store<A: BatchMatmulRoutine<(), Blueprint = BatchMatmulBlueprint>>(
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    blueprint_strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let (lhs, rhs) = tma_inputs(client, lhs, rhs)?;
    // A tensor map is read and written one element at a time.
    let vector_sizes = AvailableVectorSizes {
        out: vec![1],
        ..AvailableVectorSizes::from_type_size_tma(client, dtypes.acc_global.size())
    };
    launch_inner_ref::<TensorMapStoreArgs, A>(
        client,
        lhs,
        rhs,
        out,
        blueprint_strategy,
        vector_sizes,
        dtypes,
    )
}

/// The inputs as TMA reads them: on a device with TMA, and without permuted batches.
#[allow(clippy::result_large_err)]
fn tma_inputs(
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
) -> Result<(InputBinding, InputBinding), MatmulSetupError> {
    if !client.properties().features.tma.contains(Tma::Base) {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TmaUnavailable,
        ));
    }
    let contiguous_batches = |binding: InputBinding| match matrix_batch_layout(
        &binding.data().strides,
        binding.scheme(),
    ) {
        MatrixBatchLayout::Contiguous
        | MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: false,
        } => Ok(binding),
        MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: true,
        }
        | MatrixBatchLayout::HighlyPermuted => binding.into_contiguous(client),
    };
    Ok((contiguous_batches(lhs)?, contiguous_batches(rhs)?))
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<MA: MatmulArgs<Config = ()>, A: BatchMatmulRoutine<()>>(
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
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

    launch_kernel_concrete::<MA, A>(
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
