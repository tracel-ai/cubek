//! Materialize convolution windows directly into a strided column matrix.

use cubecl::{calculate_cube_count_elemwise, prelude::*, std::FastDivmod};

use crate::{ConvolutionArgs, components::ConvSetupError};

/// An NHWC input (with one to three spatial axes) and `[batch * out_spatial, taps * C]` output.
pub struct Im2colTensors {
    pub input: TensorBinding,
    pub columns: TensorBinding,
}

/// Gather F32 convolution windows, including padding, with one write per output element.
///
/// Both bindings may be strided. The output's row pitch is preserved, so callers can
/// pass the matrix straight to matmul without an intermediate reshape copy.
pub fn launch_im2col<const N: usize>(
    client: &Client,
    tensors: Im2colTensors,
    out_shape: [usize; N],
    kernel_shape: [usize; N],
    args: ConvolutionArgs<N>,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    let Im2colTensors { input, columns } = tensors;
    if dtype != f32::elem_type_native()
        || !(1..=3).contains(&N)
        || input.shape.len() != N + 2
        || columns.shape.len() != 2
        || input.tiling.is_tiled()
        || columns.tiling.is_tiled()
        || args.stride.contains(&0)
        || args.dilation.contains(&0)
        || kernel_shape.contains(&0)
    {
        return Err(ConvSetupError::Unknown);
    }
    let channels = input.shape[N + 1];
    let rows = out_shape
        .iter()
        .try_fold(input.shape[0], |n, d| n.checked_mul(*d));
    let width = kernel_shape
        .iter()
        .try_fold(channels, |n, d| n.checked_mul(*d));
    let (Some(rows), Some(width)) = (rows, width) else {
        return Err(ConvSetupError::Unknown);
    };
    if columns.shape[0] != rows || columns.shape[1] != width {
        return Err(ConvSetupError::Unknown);
    }
    let Some(elements) = rows.checked_mul(width) else {
        return Err(ConvSetupError::Unknown);
    };
    if elements == 0 {
        return Ok(());
    }
    let vector_size = if input.strides[N + 1] == 1 && columns.strides[1] == 1 {
        client
            .io_optimized_vector_sizes(dtype.size())
            .filter(|v| *v <= 4)
            .find(|v| {
                channels.is_multiple_of(*v)
                    && input.strides[..N + 1].iter().all(|s| s.is_multiple_of(*v))
                    && columns.strides[0].is_multiple_of(*v)
            })
            .unwrap_or(1)
    } else {
        1
    };
    let mut spatial = SequenceArg::<FastDivmod<usize>>::new();
    let mut kernel = SequenceArg::<FastDivmod<usize>>::new();
    let mut stride = SequenceArg::new();
    let mut padding = SequenceArg::new();
    let mut dilation = SequenceArg::new();
    for axis in 0..N {
        spatial.push(out_shape[axis]);
        kernel.push(kernel_shape[axis]);
        stride.push(args.stride[axis]);
        padding.push(args.padding[axis]);
        dilation.push(args.dilation[axis]);
    }
    let units = elements / vector_size;
    let cube_dim = CubeDim::new(client, units);
    let address_type = input
        .required_address_type(dtype.size())
        .max(columns.required_address_type(dtype.size()));
    im2col_kernel::launch(
        client,
        calculate_cube_count_elemwise(client, units, cube_dim),
        cube_dim,
        address_type,
        vector_size,
        input.into_tensor_arg(),
        columns.into_tensor_arg(),
        width,
        channels,
        spatial,
        kernel,
        stride,
        padding,
        dilation,
    );
    Ok(())
}

#[cube(launch, address_type = "dynamic")]
fn im2col_kernel<V: Size>(
    input: &Tensor<Vector<f32, V>>,
    columns: &mut Tensor<Vector<f32, V>>,
    width: FastDivmod<usize>,
    channels: FastDivmod<usize>,
    spatial: Sequence<FastDivmod<usize>>,
    kernel: Sequence<FastDivmod<usize>>,
    stride: Sequence<usize>,
    padding: Sequence<usize>,
    dilation: Sequence<usize>,
) {
    let vector_size = columns.vector_size();
    let (row, column) = width.div_mod(ABSOLUTE_POS * vector_size);
    if row >= columns.shape(0) {
        terminate!();
    }
    let dims = spatial.len().comptime();
    let (mut tap, channel) = channels.div_mod(column);
    let mut batch = row;
    let mut offset = channel * input.stride(dims + 1);
    let mut valid = true;
    #[unroll]
    for rev in 0..dims {
        let axis = dims - rev - 1;
        let (remaining, position) = spatial[axis].div_mod(batch);
        batch = remaining;
        let (remaining_tap, kernel_position) = kernel[axis].div_mod(tap);
        tap = remaining_tap;
        let padded_position = position * stride[axis] + kernel_position * dilation[axis];
        if padded_position >= padding[axis] {
            let input_position = padded_position - padding[axis];
            if input_position < input.shape(axis + 1) {
                offset += input_position * input.stride(axis + 1);
            } else {
                valid = false;
            }
        } else {
            valid = false;
        }
    }
    offset += batch * input.stride(0);
    let mut value = Vector::<f32, V>::new(0.0);
    if valid {
        value = input[offset / vector_size];
    }
    let output_offset = row * columns.stride(0) + column * columns.stride(1);
    columns[output_offset / vector_size] = value;
}
