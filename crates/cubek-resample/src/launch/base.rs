use crate::{components::resample_kernel, definition::Resample};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{
        FastDivmod,
        tensor::{
            launch::ViewArg,
            layout::{
                CoordsDyn,
                fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
            },
        },
    },
    tensor_vector_size_parallel,
};

/// Launch the resample kernel for a single spatial axis.
pub fn resample_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    config: Resample,
    dtype: StorageType,
) {
    let supported_vector_sizes = client.io_optimized_vector_sizes(dtype.size());
    let mut vector_size = 1;

    for vs in supported_vector_sizes {
        let vs_in = tensor_vector_size_parallel(
            core::iter::once(vs),
            &input.shape,
            &input.strides,
            input.shape.len() - 1,
        );
        let vs_out = tensor_vector_size_parallel(
            core::iter::once(vs),
            &output.shape,
            &output.strides,
            output.shape.len() - 1,
        );

        if vs_in == vs && vs_out == vs {
            vector_size = vs;
            break;
        }
    }

    let vectorized_axis = input.shape.len() - 1;

    let working_units = output.shape.iter().product::<usize>() / vector_size;

    let cube_dim = CubeDim::new(&client, working_units);

    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);

    let output_shape = divmod_sequence(&output.shape);
    let output_strides = divmod_sequence(&output.strides);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            vector_size,
            view(input, vector_size),
            view(output, vector_size),
            output_shape,
            output_strides,
            working_units,
            config,
            vectorized_axis,
            dtype,
        );
    }
}

fn divmod_sequence<R: Runtime>(shape: &[usize]) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

fn view<R: Runtime>(tensor: TensorBinding<R>, vector_size: VectorSize) -> ViewArg<CoordsDyn, R> {
    let shape_seq = tensor
        .shape
        .iter()
        .map(|&s| s as u32)
        .collect::<SequenceArg<R, u32>>();

    let layout = FixedDimLayoutLaunch::<CoordsDyn, R>::from_shape_handle_unchecked(
        &tensor,
        shape_seq,
        vector_size,
    );
    let buffer = tensor.into_tensor_arg();
    ViewArg::new_tensor::<FixedDimLayout<CoordsDyn>>(buffer, layout)
}
