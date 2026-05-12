use super::get_ratio;
use crate::{InterpolateError, kernel::forward::get_pixel_fraction};
use cubecl::{calculate_cube_count_elemwise, prelude::*, tensor_vector_size_parallel};

#[cube(launch_unchecked, address_type = "dynamic")]
fn interpolate_nearest_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    scale_h: f32,
    scale_w: f32,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;

    let vector_size = input.vector_size();

    let c_dim = output.shape(3) / vector_size;
    let w_dim = output.shape(2);
    let h_dim = output.shape(1);

    let mut temp_idx = out_idx;

    let c_idx = temp_idx % c_dim;
    temp_idx /= c_dim;

    let x_out = temp_idx % w_dim;
    temp_idx /= w_dim;

    let y_out = temp_idx % h_dim;
    let batch = temp_idx / h_dim;

    let h_in = input.shape(1) as f32;
    let w_in = input.shape(2) as f32;

    let y_in_f = get_pixel_fraction(y_out, scale_h, false)
        .floor()
        .clamp(0.0, h_in - 1.0);

    let x_in_f = get_pixel_fraction(x_out, scale_w, false)
        .floor()
        .clamp(0.0, w_in - 1.0);

    let y_in = usize::cast_from(y_in_f);
    let x_in = usize::cast_from(x_in_f);

    let in_idx = (batch * input.stride(0) + y_in * input.stride(1) + x_in * input.stride(2))
        / vector_size
        + c_idx;

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

    let scale_h = get_ratio(input.shape[1], output.shape[1], false);
    let scale_w = get_ratio(input.shape[2], output.shape[2], false);

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
            scale_h,
            scale_w,
            dtype,
        )
    };

    Ok(())
}
