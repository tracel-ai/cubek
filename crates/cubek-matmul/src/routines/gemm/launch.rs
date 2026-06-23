use cubecl::{
    zspace::Shape,
    {VectorizationError, prelude::*},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    components::batch::gemm::{MatmulOperandLayouts, OperandLayout},
    definition::cube_mapping_launch,
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
};

use crate::{
    args::InputArg,
    args::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs},
    routines::gemm::GemmRoutine,
    routines::{BatchMatmulRoutine as _, BlueprintStrategy, into_contiguous_if_highly_permuted},
};

fn vector_size_for<R: Runtime>(
    client: &ComputeClient<R>,
    binding: &InputBinding<R>,
    default_size: usize,
    plane_size: usize,
    dim: usize,
) -> Result<usize, VectorizationError> {
    let (size, num_quants) = if let InputBinding::Quantized { scheme, .. } = binding {
        (scheme.size_bits_stored() / 8, scheme.num_quants())
    } else {
        (default_size, 1)
    };
    client
        .io_optimized_vector_sizes(size)
        .filter(|&v| dim.is_multiple_of(plane_size * v * num_quants))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)
}

/// Return `binding` materialized into the `target` matrix layout, copying the
/// data only when its current layout differs. `RowMajor` leaves the last dim
/// contiguous, `ColMajor` the second-to-last.
#[allow(clippy::result_large_err)]
fn make_k_contiguous<R: Runtime>(
    client: &ComputeClient<R>,
    binding: InputBinding<R>,
    target: MatrixLayout,
) -> Result<InputBinding<R>, MatmulSetupError> {
    let rank = binding.shape().len();
    let layout = MatrixLayout::from_shape_and_strides(
        binding.shape(),
        &binding.data().strides,
        binding.scheme(),
    )?;
    if layout == target {
        return Ok(binding);
    }
    Ok(match target {
        MatrixLayout::RowMajor => binding.into_contiguous(client)?,
        MatrixLayout::ColMajor => {
            let mut binding = binding;
            binding.swap_dims(rank - 2, rank - 1);
            let mut binding = binding.into_contiguous(client)?;
            binding.swap_dims(rank - 2, rank - 1);
            binding
        }
    })
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    mut lhs: InputBinding<R>,
    mut rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), GemmRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    // A stride-0 (broadcast) matrix dim can't be classified as row/col-major;
    // materialize such operands so they read as plain contiguous tensors.
    lhs = into_contiguous_if_highly_permuted(client, lhs)?;
    rhs = into_contiguous_if_highly_permuted(client, rhs)?;

    // On GPU only the `Dot` variant has a kernel, and it needs both operands
    // K-contiguous, so normalize each float matrix operand to that layout.
    //
    // Quantized operands are excluded here by the `scheme().is_none()` guards. A
    // quantized operand that is not already K-contiguous can only reach `Dot` by
    // being transposed, and that transpose repacks its u32 buffer through cubecl
    // `into_contiguous_packed`, which currently writes a zero buffer. Quantized
    // matmul on GPU is therefore BLOCKED on that cubecl bug being fixed upstream.
    // When it lands, remove these guards so quantized operands take the same path.
    let plane_dim = client.properties().hardware.plane_size_max as usize;
    if plane_dim > 1 {
        let rank = lhs.shape().len();
        let m = lhs.shape().to_vec()[rank - 2];
        let n = rhs.shape().to_vec()[rank - 1];
        if m > 1 && lhs.scheme().is_none() {
            lhs = make_k_contiguous(client, lhs, MatrixLayout::RowMajor)?;
        }
        if n > 1 && rhs.scheme().is_none() {
            rhs = make_k_contiguous(client, rhs, MatrixLayout::ColMajor)?;
        }
    }

    let rank = rhs.shape().len();
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    let m = lhs_shape.to_vec()[rank - 2];
    let n = rhs_shape.to_vec()[rank - 1];
    let k = lhs_shape.to_vec()[rank - 1];

    let plane_size = client.properties().hardware.plane_size_max as usize;

    // For variants that walk K with vector-size steps, k must be divisible
    // by plane_size to even have a valid vector_size — the Family's
    // validate_blueprint enforces the full per-variant divisibility.
    if !k.is_multiple_of(plane_size) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Dimension k={} must be a multiple of plane size {}",
            k, plane_size
        ))));
    }

    let lhs_vector_size = vector_size_for(client, &lhs, dtypes.lhs_global.size(), plane_size, k)?;
    let rhs_vector_size = vector_size_for(client, &rhs, dtypes.rhs_global.size(), plane_size, k)?;

    let shared_vector_size = lhs_vector_size.min(rhs_vector_size);

    let vector_sizes = MatmulVectorSizes {
        lhs: shared_vector_size,
        rhs: shared_vector_size,
        out: 1,
    };

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let lhs_batches: Shape = lhs.shape().to_vec()[..rank - 2].into();
    let rhs_batches: Shape = rhs.shape().to_vec()[..rank - 2].into();

    let lhs_layout =
        MatrixLayout::from_shape_and_strides(lhs_shape, &lhs.data().strides, lhs.scheme())?;
    let rhs_layout =
        MatrixLayout::from_shape_and_strides(rhs_shape, &rhs.data().strides, rhs.scheme())?;

    let kind_problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches.clone(),
        rhs_batches.clone(),
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let kind = MatmulOperandLayouts::from_problem(&kind_problem)?;

    // Vec operands need K-contiguous storage; the kernel reads them as if
    // they were the side that supplies K-vectors / scalars. Mat operands
    // keep their natural layout — the Family picks the variant from
    // `MatmulOperandLayouts`.
    if matches!(kind.lhs, OperandLayout::Vector) && kind_problem.lhs_strides[rank - 1] != 1 {
        lhs = lhs.into_contiguous(client)?;
    }
    if matches!(kind.rhs, OperandLayout::Vector) && kind_problem.rhs_strides[rank - 1] != 1 {
        rhs = rhs.into_contiguous(client)?;
    }

    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let device_settings = GemmRoutine::device_settings(client, vector_sizes);
    let expand_info = GemmRoutine::expand_blueprint(&problem, &device_settings, strategy)?;
    let launch_info = GemmRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<GemmRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<GemmRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );

    GemmRoutine::launch::<TensorArgs, R>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        launch_info.address_type,
        input,
        output,
        (),
        cube_mapping_launch(&launch_info.cube_count_plan),
        launch_info.blueprint,
        dtypes,
        &launch_info.vector_sizes,
    )
}
