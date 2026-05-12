use super::super::shape_divmod;
use crate::InterpolateError;
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::FastDivmod, tensor_vector_size_parallel,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn interpolate_nearest_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;

    let vector_size = input.vector_size();

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let ratio_h = (input.shape(1) as f32) / (output.shape(1) as f32);
    let ratio_w = (input.shape(2) as f32) / (output.shape(2) as f32);

    let y_in = usize::cast_from(((y as f32 + 0.5) * ratio_h - 0.5).floor());
    let x_in = usize::cast_from(((x as f32 + 0.5) * ratio_w - 0.5).floor());

    let y_in = y_in.clamp(0, input.shape(1) - 1);
    let x_in = x_in.clamp(0, input.shape(2) - 1);

    let in_idx =
        (b * input.stride(0) + y_in * input.stride(1) + x_in * input.stride(2)) / vector_size + c;

    output[out_idx] = input[in_idx];
}

pub(crate) fn interpolate_nearest_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );
    let out_shape = shape_divmod(&output);

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        interpolate_nearest_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            out_shape,
            dtype,
        )
    };

    Ok(())
}
