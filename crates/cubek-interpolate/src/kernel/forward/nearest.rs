use crate::InterpolateError;
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

    let vec_size = input.vector_size();
    let c_dim = output.shape(3) / vec_size;
    let w_dim = output.shape(2);
    let h_dim = output.shape(1);

    let c_idx = out_idx % c_dim;
    let rem = out_idx / c_dim;
    let x_out = rem % w_dim;
    let rem = rem / w_dim;
    let y_out = rem % h_dim;
    let batch = rem / h_dim;

    let y_in = usize::cast_from(((y_out as f32 + 0.5) * scale_h - 0.5).floor());
    let x_in = usize::cast_from(((x_out as f32 + 0.5) * scale_w - 0.5).floor());

    let y_in = y_in.clamp(0, input.shape(1) - 1);
    let x_in = x_in.clamp(0, input.shape(2) - 1);

    let in_idx = (batch * input.stride(0) + y_in * input.stride(1) + x_in * input.stride(2))
        / vec_size
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

    let scale_h = (input.shape[1] as f32) / (output.shape[1] as f32);
    let scale_w = (input.shape[2] as f32) / (output.shape[2] as f32);

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
