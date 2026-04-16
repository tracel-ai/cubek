use cubecl::{
    std::tensor::{MatrixBatchLayout, matrix_batch_layout},
    zspace::Shape,
    {VectorizationError, prelude::*},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    definition::{MatmulVectorSizes, cube_mapping_launch},
};

use crate::{
    launch::InputArg,
    launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs},
    routines::vecmat_unit_perpendicular::GemvUnitPerpendicularRoutine,
    routines::{BlueprintStrategy, Routine as _},
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

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), GemvUnitPerpendicularRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = rhs.shape().len();

    // Rhs is assumed row major for now
    let rhs_layout = matrix_batch_layout(&rhs.data().strides, rhs.scheme());
    let rhs = if !matches!(rhs_layout, MatrixBatchLayout::Contiguous) {
        rhs.into_contiguous(client)?
    } else {
        rhs
    };

    let m = lhs.shape().to_vec()[rank - 2];
    let n = rhs.shape().to_vec()[rank - 1];
    let k = lhs.shape().to_vec()[rank - 1];

    if m != 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "m must equal 1 to qualify as a vecmat problem",
        )));
    }

    let rhs_shape = rhs.shape();

    let plane_size = client.properties().hardware.plane_size_max as usize;

    if !k.is_multiple_of(plane_size) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Lhs dimension k={} must be a multiple of plane size {}",
            k, plane_size
        ))));
    }

    if !n.is_multiple_of(plane_size) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Rhs dimension n={} must be a multiple of plane size {}",
            n, plane_size
        ))));
    }

    let lhs_vector_size = vector_size_for(client, &lhs, dtypes.lhs_global.size(), plane_size, k)?;
    // Assumes rhs is row major
    let rhs_vector_size = vector_size_for(client, &rhs, dtypes.rhs_global.size(), plane_size, n)?;

    let shared_vector_size = lhs_vector_size.min(rhs_vector_size);

    let vector_sizes = MatmulVectorSizes {
        lhs: shared_vector_size,
        rhs: shared_vector_size,
        out: shared_vector_size,
    };

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let lhs_batches: Shape = lhs.shape().to_vec()[..rank - 2].into();
    let rhs_batches: Shape = rhs.shape().to_vec()[..rank - 2].into();

    let problem = MatmulProblem::from_parameters(
        1,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::from_shape_and_strides(rhs_shape, &rhs.data().strides, rhs.scheme())?,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    if problem.rhs_layout != MatrixLayout::RowMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Vecmat unit perpendicular only supports row major rhs for now",
        )));
    }

    let device_settings = GemvUnitPerpendicularRoutine::device_settings(client, vector_sizes);
    let expand_info =
        GemvUnitPerpendicularRoutine::expand_blueprint(&problem, &device_settings, strategy)?;
    let launch_info =
        GemvUnitPerpendicularRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input =
        <InputArg<TensorArgs> as ConcreteInputsFactory<GemvUnitPerpendicularRoutine>>::create(
            lhs,
            rhs,
            &launch_info.blueprint,
            &problem,
            &vector_sizes,
            dtypes,
        );
    let output =
        <OutputArg<TensorArgs> as ConcreteOutputFactory<GemvUnitPerpendicularRoutine>>::create(
            out,
            &launch_info.blueprint,
            &problem,
            &vector_sizes,
            dtypes,
        );

    GemvUnitPerpendicularRoutine::launch::<TensorArgs, R>(
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
