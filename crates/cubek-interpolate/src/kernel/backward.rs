use crate::definition::NearestMode;
use cubecl::prelude::*;
use cubecl::{
    calculate_cube_count_elemwise,
    client::ComputeClient,
    ir::UIntKind,
    num_traits::Zero,
    prelude::TensorBinding,
    std::FastDivmod,
    std::tensor::layout::{
        linear::{LinearLayout, LinearLayoutLaunch, LinearViewLayoutLaunch},
        *,
    },
    tensor_vector_size_parallel,
};

pub(crate) fn interpolate_nearest_backward_launch<R: Runtime>(
    client: &ComputeClient<R>,
    out_grad: TensorBinding<R>,
    input_grad: TensorBinding<R>,
    nearest_mode: NearestMode,
    dtype: ElemType,
) -> Result<(), crate::definition::InterpolateError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &out_grad.shape,
        &out_grad.strides,
        out_grad.shape.len() - 1,
    );
    let input_grad_shape = shape_divmod(&input_grad);
    let input_grad_layout = linear_layout(&input_grad, vector_size);
    let working_units = input_grad.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let address_type = out_grad
        .required_address_type(dtype.size())
        .max(input_grad.required_address_type(dtype.size()));

    unsafe {
        execute_interpolate_nearest_backward::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            input_grad.clone().into_tensor_arg(),
            input_grad_shape,
            input_grad_layout,
            nearest_mode,
            dtype,
        )
    };
    Ok(())
}

fn shape_divmod<R: Runtime>(binding: &TensorBinding<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

fn linear_layout<R: Runtime>(
    binding: &TensorBinding<R>,
    vector_size: usize,
) -> LinearLayoutLaunch<R> {
    LinearLayoutLaunch::from_shape_strides(
        binding.shape.clone(),
        binding.strides.clone(),
        Type::new(ElemType::UInt(UIntKind::U32)).with_vector_size(vector_size),
        LinearViewLayoutLaunch::new(),
    )
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn execute_interpolate_nearest_backward<F: Float, N: Size>(
    grad: &Tensor<Vector<F, N>>,
    input_grad: &mut Tensor<Vector<F, N>>,
    input_grad_shape: Sequence<FastDivmod<usize>>,
    input_grad_layout: LinearLayout,
    #[comptime] nearest_mode: NearestMode,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= input_grad.len() {
        terminate!();
    }

    let vector_size = grad.vector_size();
    let input_grad_idx = input_grad_layout.to_source_pos(ABSOLUTE_POS);

    let out_h = input_grad.shape(1);
    let out_w = input_grad.shape(2);
    let grad_h = grad.shape(1);
    let grad_w = grad.shape(2);

    let (rem, c) = input_grad_shape[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, out_x) = input_grad_shape[2].div_mod(rem);
    let (b, out_y) = input_grad_shape[1].div_mod(rem);

    let grad_y_start = start_index::<F>(out_y, grad_h, out_h, nearest_mode);
    let grad_y_end = end_index::<F>(out_y, grad_h, out_h, nearest_mode);
    let grad_x_start = start_index::<F>(out_x, grad_w, out_w, nearest_mode);
    let grad_x_end = end_index::<F>(out_x, grad_w, out_w, nearest_mode);

    let index_grad_base = b * grad.stride(0) + c * grad.stride(3);

    let mut sum = Vector::zero();

    for grad_y in grad_y_start..grad_y_end {
        for grad_x in grad_x_start..grad_x_end {
            let index_grad = index_grad_base + grad_y * grad.stride(1) + grad_x * grad.stride(2);

            sum += grad[index_grad / vector_size];
        }
    }

    input_grad[input_grad_idx] = sum;
}

#[cube]
fn start_index<F: Float>(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    #[comptime] nearest_mode: NearestMode,
) -> usize {
    match nearest_mode {
        NearestMode::Floor => {
            let numerator = F::cast_from(input_index * output_size);
            let div = (numerator / F::cast_from(input_size)).ceil();
            usize::cast_from(div)
        }
        NearestMode::Exact => {
            let num = F::cast_from(input_index * output_size);
            let den = F::cast_from(input_size);
            let div = num / den - F::new(0.5_f32);

            let mask = F::cast_from((div >= F::zero()) as usize);
            usize::cast_from(div.ceil() * mask)
        }
    }
}

#[cube]
fn end_index<F: Float>(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    #[comptime] nearest_mode: NearestMode,
) -> usize {
    start_index::<F>(input_index + 1, output_size, input_size, nearest_mode)
}
