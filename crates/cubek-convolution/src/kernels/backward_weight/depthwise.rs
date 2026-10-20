//! Spatially split weight gradients for depthwise 2D convolutions.

use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::TensorHandle};

use crate::{ConvolutionArgs, components::ConvSetupError};

const UNITS: u32 = 128;
const SAMPLES_PER_CUBE: usize = 4096;
const MAX_CHANNEL_PARTIAL_BYTES: usize = 128 * 1024 * 1024;

/// NHWC input and output gradient, and `[C_out, kh, kw, 1]` weight gradient.
pub struct DepthwiseWeightBackwardTensors {
    pub input: TensorBinding,
    pub out_grad: TensorBinding,
    pub weight_grad: TensorBinding,
}

/// Compute F32 depthwise weight gradients, including channel multipliers.
///
/// Each cube reduces a bounded piece of the batch/spatial domain. A second
/// kernel sums those pieces, without atomics or reordering either input. All
/// bindings may be strided. Other dtypes and non-depthwise convolutions are
/// declined so callers can select another algorithm.
pub fn launch_depthwise_weight_backward(
    client: &Client,
    tensors: DepthwiseWeightBackwardTensors,
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    let DepthwiseWeightBackwardTensors {
        input,
        out_grad,
        weight_grad,
    } = tensors;
    let hardware = &client.properties().hardware;
    if dtype != f32::elem_type_native()
        || hardware.max_units_per_cube < UNITS
        || hardware.max_shared_memory_size < UNITS as usize * 4
        || input.shape.len() != 4
        || out_grad.shape.len() != 4
        || weight_grad.shape.len() != 4
        || input.tiling.is_tiled()
        || out_grad.tiling.is_tiled()
        || weight_grad.tiling.is_tiled()
    {
        return Err(ConvSetupError::Unknown);
    }
    let channels = input.shape[3];
    let out_channels = out_grad.shape[3];
    if channels == 0
        || groups != channels
        || out_channels == 0
        || !out_channels.is_multiple_of(channels)
        || weight_grad.shape[0] != out_channels
        || weight_grad.shape[3] != 1
        || input.shape[0] != out_grad.shape[0]
        || args.stride.contains(&0)
        || args.dilation.contains(&0)
    {
        return Err(ConvSetupError::Unknown);
    }

    let weights = weight_grad.shape.iter().product::<usize>();
    if weights == 0 {
        return Ok(());
    }
    let samples = out_grad.shape[..3].iter().product::<usize>();
    // With NHWC storage, adjacent units should read adjacent channels. Bound
    // the larger partial buffer this uses; other layouts and larger problems
    // keep reducing one filter per cube.
    let channels_per_cube = if input.strides[3] == 1
        && out_grad.strides[3] == 1
        && out_channels >= 32
        && samples
            .div_ceil(SAMPLES_PER_CUBE / 32)
            .max(1)
            .checked_mul(weights)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .is_some_and(|bytes| bytes <= MAX_CHANNEL_PARTIAL_BYTES)
    {
        32
    } else {
        1
    };
    let splits = samples
        .div_ceil(SAMPLES_PER_CUBE / channels_per_cube)
        .max(1);
    let partials = TensorHandle::empty(client, [splits, weights], dtype);
    let address_type = input
        .required_address_type(dtype.size())
        .max(out_grad.required_address_type(dtype.size()))
        .max(weight_grad.required_address_type(dtype.size()))
        .max(partials.required_address_type());
    let cube_dim = CubeDim::new_1d(UNITS);
    let kh = weight_grad.shape[1];
    let kw = weight_grad.shape[2];
    let weight_blocks = kh * kw * out_channels.div_ceil(channels_per_cube);
    let cube_count =
        calculate_cube_count_elemwise(client, splits * weight_blocks * UNITS as usize, cube_dim);

    depthwise_weight_partials::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        input.into_tensor_arg(),
        out_grad.into_tensor_arg(),
        partials.clone().into_arg(),
        weight_blocks,
        samples,
        kh,
        kw,
        out_channels / channels,
        args.stride[0],
        args.stride[1],
        args.padding[0],
        args.padding[1],
        args.dilation[0],
        args.dilation[1],
        channels_per_cube,
    );
    depthwise_weight_finish::launch(
        client,
        calculate_cube_count_elemwise(client, weights, cube_dim),
        cube_dim,
        address_type,
        partials.into_arg(),
        weight_grad.into_tensor_arg(),
        weights,
    );
    Ok(())
}

#[cube(launch, address_type = "dynamic")]
fn depthwise_weight_partials(
    input: &Tensor<f32>,
    grad: &Tensor<f32>,
    partials: &mut Tensor<f32>,
    weight_blocks: usize,
    samples: usize,
    kh: usize,
    kw: usize,
    multiplier: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    #[comptime] channels_per_cube: usize,
) {
    let block = CUBE_POS;
    let split = block / weight_blocks;
    if split >= partials.shape(0) {
        terminate!();
    }
    let weight = block % weight_blocks;
    let kx = weight % kw;
    let ky = (weight / kw) % kh;
    let oc = weight / (kh * kw) * channels_per_cube + UNIT_POS as usize % channels_per_cube;
    let ic = oc / multiplier;
    let start = split * (SAMPLES_PER_CUBE / channels_per_cube);
    let end = clamp_max(start + SAMPLES_PER_CUBE / channels_per_cube, samples);
    let mut sum = 0.0f32;
    if oc < grad.shape(3) {
        let mut row = start + UNIT_POS as usize / channels_per_cube;
        while row < end {
            let ox = row % grad.shape(2);
            let oy = (row / grad.shape(2)) % grad.shape(1);
            let batch = row / (grad.shape(1) * grad.shape(2));
            let iy = oy * sh + ky * dh;
            let ix = ox * sw + kx * dw;
            if iy >= ph && ix >= pw && iy - ph < input.shape(1) && ix - pw < input.shape(2) {
                let input_idx = batch * input.stride(0)
                    + (iy - ph) * input.stride(1)
                    + (ix - pw) * input.stride(2)
                    + ic * input.stride(3);
                let grad_idx = batch * grad.stride(0)
                    + oy * grad.stride(1)
                    + ox * grad.stride(2)
                    + oc * grad.stride(3);
                sum += input[input_idx] * grad[grad_idx];
            }
            row += UNITS as usize / channels_per_cube;
        }
    }

    let mut shared = Shared::<[f32]>::new_slice(UNITS as usize);
    shared[UNIT_POS as usize] = sum;
    sync_cube();
    // The cube size is fixed, so each reduction level has a constant offset.
    #[unroll]
    for level in 0..(UNITS as usize / channels_per_cube).ilog2() {
        let active = UNITS >> (level + 1);
        if UNIT_POS < active {
            let other = shared[(UNIT_POS + active) as usize];
            shared[UNIT_POS as usize] += other;
        }
        sync_cube();
    }
    if (UNIT_POS as usize) < channels_per_cube && oc < grad.shape(3) {
        let weight_index = (oc * kh + ky) * kw + kx;
        let offset = split * partials.stride(0) + weight_index * partials.stride(1);
        partials[offset] = shared[UNIT_POS as usize];
    }
}

#[cube(launch, address_type = "dynamic")]
fn depthwise_weight_finish(partials: &Tensor<f32>, grad: &mut Tensor<f32>, weights: usize) {
    let weight = ABSOLUTE_POS;
    if weight >= weights {
        terminate!();
    }
    let mut sum = 0.0f32;
    for split in 0..partials.shape(0) {
        sum += partials[split * partials.stride(0) + weight * partials.stride(1)];
    }
    let kw = grad.shape(2);
    let kh = grad.shape(1);
    let kx = weight % kw;
    let ky = (weight / kw) % kh;
    let oc = weight / (kh * kw);
    let offset = oc * grad.stride(0) + ky * grad.stride(1) + kx * grad.stride(2);
    grad[offset] = sum;
}
