use super::shape_divmod;
use crate::InterpolateError;
use cubecl::std::FastDivmod;
use cubecl::{calculate_cube_count_elemwise, prelude::*, tensor_vector_size_parallel};

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

    let vector_size = input.vector_size();
    let out_idx = ABSOLUTE_POS;

    let out_pos = ABSOLUTE_POS * vector_size;

    let (h_in, w_in) = (input.shape(1), input.shape(2));
    let (h_out, w_out) = (output.shape(1), output.shape(2));

    let (rem, c) = shape_out[3].div_mod(out_pos);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let y = y * h_in / h_out;
    let x = x * w_in / w_out;

    let in_idx =
        b * input.stride(0) + y * input.stride(1) + x * input.stride(2) + c * input.stride(3);

    output[out_idx] = input[in_idx / vector_size];
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

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let shape_out = shape_divmod(&output);
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
            shape_out,
            dtype,
        )
    };

    Ok(())
}
