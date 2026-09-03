use super::pool3d::{Position3d, view5d};
use crate::{
    definition::PoolError,
    kernel::{
        accumulator_dtype, adaptive_end_index as end_index, adaptive_start_index as start_index,
        adaptive_window_address_type, decompose_linear_5d, shape_divmod,
    },
};
use cubecl::{
    CubeDim, Runtime, calculate_cube_count_elemwise,
    num_traits::Zero,
    prelude::{TensorBinding, *},
    std::{FastDivmod, tensor::ViewMut},
    tensor_vector_size_parallel,
};

#[cube(launch, address_type = "dynamic")]
fn adaptive_avg_pool3d_direct<EI: Float, EA: Float, N: Size>(
    input: &Tensor<Vector<EI, N>>,
    mut output: ViewMut<'_, Vector<EI, N>, Position3d>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(EI)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (b, od, oh, ow, c) = decompose_linear_5d(ABSOLUTE_POS * output.vector_size(), &out_shape);
    let (_, out_d, out_h, out_w, _) = output.shape();
    let (in_d, in_h, in_w) = (input.shape(1), input.shape(2), input.shape(3));

    let id_start = start_index(od, out_d, in_d);
    let id_end = end_index(od, out_d, in_d);
    let ih_start = start_index(oh, out_h, in_h);
    let ih_end = end_index(oh, out_h, in_h);
    let iw_start = start_index(ow, out_w, in_w);
    let iw_end = end_index(ow, out_w, in_w);

    let mut sum = Vector::<EA, N>::zero();
    let index_input_base = b * input.stride(0) + c * input.stride(4);

    for id in id_start..id_end {
        let index_input_d = id * input.stride(1);
        for ih in ih_start..ih_end {
            let index_input_h = ih * input.stride(2);
            for iw in iw_start..iw_end {
                let index_input =
                    index_input_base + index_input_d + index_input_h + iw * input.stride(3);
                sum += Vector::cast_from(input[index_input / input.vector_size()]);
            }
        }
    }

    let volume = (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);
    let average = sum / Vector::cast_from(volume);
    output.write((b, od, oh, ow, c), Vector::cast_from(average));
}

pub(crate) fn adaptive_avg_pool3d_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: ElemType,
) -> Result<(), PoolError> {
    let acc_dtype = accumulator_dtype(dtype);
    let input_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );
    let output_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        output.shape.len() - 1,
    );
    let vector_size = input_vector_size.min(output_vector_size);
    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;

    if working_units == 0 {
        return Ok(());
    }

    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()))
        .max(adaptive_window_address_type(
            &input.shape[1..4],
            &output.shape[1..4],
        ));

    adaptive_avg_pool3d_direct::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        input.into_tensor_arg(),
        view5d(output.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        dtype,
        acc_dtype,
    );

    Ok(())
}
