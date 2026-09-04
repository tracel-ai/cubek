use super::super::{
    accumulator_dtype, adaptive_end_index as end_index, adaptive_start_index as start_index,
    adaptive_window_address_type, decompose_linear_5d, shape_divmod,
};
use crate::definition::PoolError;
use crate::kernel::forward::{Position3d, view5d};
use cubecl::{
    CubeDim, calculate_cube_count_elemwise,
    num_traits::Zero,
    prelude::{TensorBinding, *},
    std::{FastDivmod, tensor::ViewMut},
    tensor_vector_size_parallel,
};

#[cube(launch, address_type = "dynamic")]
fn adaptive_avg_pool3d_backward_direct<EI: Float, EA: Float, N: Size>(
    grad: &Tensor<Vector<EI, N>>,
    mut output: ViewMut<'_, Vector<EI, N>, Position3d>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(EI)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, in_d, in_h, in_w, _) = output.shape();
    let (grad_d, grad_h, grad_w) = (grad.shape(1), grad.shape(2), grad.shape(3));
    let (b, id, ih, iw, c) = decompose_linear_5d(ABSOLUTE_POS * output.vector_size(), &out_shape);

    let od_start = start_index(id, in_d, grad_d);
    let od_end = end_index(id, in_d, grad_d);
    let oh_start = start_index(ih, in_h, grad_h);
    let oh_end = end_index(ih, in_h, grad_h);
    let ow_start = start_index(iw, in_w, grad_w);
    let ow_end = end_index(iw, in_w, grad_w);

    // Each input position gathers every output window that can contain it, so work items never
    // race to update the same gradient. The inverse bounds are conservative for uneven windows;
    // the containment checks below discard candidates outside the exact forward window.
    let mut grad_acc = Vector::<EA, N>::zero();
    let index_base = b * grad.stride(0) + c * grad.stride(4);

    for od in od_start..od_end {
        let id_start = start_index(od, grad_d, in_d);
        let id_end = end_index(od, grad_d, in_d);
        if id >= id_start && id < id_end {
            for oh in oh_start..oh_end {
                let ih_start = start_index(oh, grad_h, in_h);
                let ih_end = end_index(oh, grad_h, in_h);
                if ih >= ih_start && ih < ih_end {
                    for ow in ow_start..ow_end {
                        let iw_start = start_index(ow, grad_w, in_w);
                        let iw_end = end_index(ow, grad_w, in_w);
                        if iw >= iw_start && iw < iw_end {
                            let volume =
                                (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);
                            let index = index_base
                                + od * grad.stride(1)
                                + oh * grad.stride(2)
                                + ow * grad.stride(3);
                            grad_acc +=
                                Vector::<EA, N>::cast_from(grad[index / grad.vector_size()])
                                    / Vector::cast_from(volume);
                        }
                    }
                }
            }
        }
    }

    output.write((b, id, ih, iw, c), Vector::cast_from(grad_acc));
}

pub(crate) fn adaptive_avg_pool3d_backward_launch(
    client: &Client,
    out_grad: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
) -> Result<(), PoolError> {
    let acc_dtype = accumulator_dtype(dtype);
    let grad_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &out_grad.shape,
        &out_grad.strides,
        out_grad.shape.len() - 1,
    );
    let output_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        output.shape.len() - 1,
    );
    let vector_size = grad_vector_size.min(output_vector_size);

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let address_type = out_grad
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()))
        .max(adaptive_window_address_type(
            &output.shape[1..4],
            &out_grad.shape[1..4],
        ));

    adaptive_avg_pool3d_backward_direct::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        out_grad.into_tensor_arg(),
        view5d(output.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        dtype,
        acc_dtype,
    );

    Ok(())
}
