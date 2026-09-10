//! Direct convolution: one unit computes one output element of the result.
//!
//! This is the routine every CPU convolution reaches. The accelerated routines in this crate
//! contract over channels through `Cmma`/`Mma` tiles, which need a plane of 32 or 64 lanes, and a
//! CPU plane is one lane wide. So it is written straight against cubecl rather than the tile DSL,
//! the way `cubek-pool` and `cubek-reduce` are, and like [`super::depthwise`] it is its own entry
//! point rather than a `ConvAlgorithm`: with no stage hierarchy there is nothing for a blueprint
//! to size.

use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    num_traits::Zero,
    prelude::*,
    std::tensor::layout::linear::{LinearViewMut, linear_view},
    std::{FastDivmod, FastDivmodInt},
    tensor_vector_size_parallel,
};

use crate::{components::ConvSetupError, launch::ConvolutionArgs};

/// Splits a linear position into its coordinates, innermost dimension first.
#[cube]
fn decompose_linear<I: FastDivmodInt>(pos: I, shape: &Sequence<FastDivmod<I>>) -> (I, Sequence<I>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.reversed())
}

#[derive(CubeLaunch, CubeType, Clone)]
pub(crate) struct ConvParam {
    pub stride: u32,
    pub dilation: u32,
    pub padding: i32,
}

#[derive(CubeLaunch, CubeType)]
struct Conv2dArgs {
    conv_params: Sequence<ConvParam>,
    channels_per_group: u32,
}

#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::redundant_closure)]
fn direct_conv2d_kernel<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    args: Conv2dArgs,
    shape_out: Sequence<FastDivmod<u32>>,
    shape_out_c: FastDivmod<u32>,
    #[comptime] has_padding: bool,
    #[comptime] accumulate_lanes: bool,
    #[define(E)] _dtype: ElemType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let n_spatial = comptime![shape_out.len()];

    let vector_size_out = output.vector_size();
    let pos = ABSOLUTE_POS * vector_size_out;

    let in_c_per_group = weight.shape(weight.rank() - 1) as u32;

    let (rem, out_c) = shape_out_c.div_mod(pos as u32);
    let (b, spatial_pos) = decompose_linear(rem, &shape_out);

    let g = out_c / args.channels_per_group;
    let ic_start = in_c_per_group * g;

    let bias: ComptimeOption<Vector<E, NOut>> =
        bias.map(|bias| bias[out_c as usize / vector_size_out]);
    let mut sum = bias.unwrap_or_else(|| Vector::zero());

    let in_offs = b as usize * input.stride(0) + ic_start as usize;

    let stride_oc = weight.stride(0);

    let mut in_shape = Sequence::new();
    let mut in_strides = Sequence::new();
    let mut kernel_shape = Sequence::new();
    let mut kernel_strides = Sequence::new();

    #[unroll]
    for i in 0..n_spatial {
        in_shape.push(input.shape(i + 1) as u32);
        in_strides.push(input.stride(i + 1));
        kernel_shape.push(weight.shape(i + 1) as u32);
        kernel_strides.push(weight.stride(i + 1));
    }

    let weight_offs = out_c as usize * stride_oc;

    let loop_params = LoopParams {
        out_pos: spatial_pos,
        in_shape,
        in_strides,
        kernel_shape,
        kernel_strides,
        conv_params: args.conv_params,
        in_c_per_group,
        stride_oc,
    };

    kernel_loop(
        input,
        weight,
        &mut sum,
        in_offs,
        true,
        weight_offs,
        &loop_params,
        0usize,
        has_padding,
        accumulate_lanes,
    );

    output.write(ABSOLUTE_POS, sum);
}

#[derive(CubeType, Clone)]
struct LoopParams {
    out_pos: Sequence<u32>,
    in_shape: Sequence<u32>,
    in_strides: Sequence<usize>,
    kernel_shape: Sequence<u32>,
    kernel_strides: Sequence<usize>,
    conv_params: Sequence<ConvParam>,

    in_c_per_group: u32,
    stride_oc: usize,
}

#[cube]
fn kernel_loop<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    sum: &mut Vector<E, NOut>,
    in_offs: usize,
    in_bounds: bool,
    weight_offs: usize,
    params: &LoopParams,
    #[comptime] kernel_dim: usize,
    #[comptime] has_padding: bool,
    #[comptime] accumulate_lanes: bool,
) {
    if comptime![kernel_dim < params.kernel_shape.len()] {
        let out_idx = *params.out_pos.index(kernel_dim);
        let conv = params.conv_params.index(kernel_dim);
        let shape = *params.in_shape.index(kernel_dim);
        let stride = *params.in_strides.index(kernel_dim);
        let k_stride = *params.kernel_strides.index(kernel_dim);

        for pos in 0..*params.kernel_shape.index(kernel_dim) {
            let in_pos = (out_idx * conv.stride + pos * conv.dilation) as i32 - conv.padding;
            let in_offs = in_offs + in_pos as usize * stride;
            let weight_offs = weight_offs + pos as usize * k_stride;
            let mut in_bounds = in_bounds;

            if has_padding {
                in_bounds &= in_pos >= 0 && (in_pos as u32) < shape;
            }

            kernel_loop(
                input,
                weight,
                sum,
                in_offs,
                in_bounds,
                weight_offs,
                params,
                comptime![kernel_dim + 1],
                has_padding,
                accumulate_lanes,
            );
        }
    } else {
        kernel_loop_inner(
            input,
            weight,
            sum,
            in_offs,
            in_bounds,
            weight_offs,
            params.in_c_per_group,
            params.stride_oc,
            accumulate_lanes,
        );
    }
}

#[cube]
fn kernel_loop_inner<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    sum: &mut Vector<E, NOut>,
    in_offs: usize,
    in_bounds: bool,
    weight_offs: usize,
    in_c_per_group: u32,
    stride_oc: usize,
    #[comptime] accumulate_lanes: bool,
) {
    if in_bounds {
        if accumulate_lanes {
            accumulate_in_lanes(
                input,
                weight,
                sum,
                in_offs,
                weight_offs,
                in_c_per_group,
                stride_oc,
            );
        } else {
            accumulate_per_step(
                input,
                weight,
                sum,
                in_offs,
                weight_offs,
                in_c_per_group,
                stride_oc,
            );
        }
    }
}

/// One input read per output channel buys a channel loop with no dependency chain.
#[cube]
fn accumulate_in_lanes<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    sum: &mut Vector<E, NOut>,
    in_offs: usize,
    weight_offs: usize,
    in_c_per_group: u32,
    stride_oc: usize,
) {
    let vector_size_in = input.vector_size();
    let vector_size_out = sum.vector_size();

    #[unroll]
    for v in 0..vector_size_out {
        let mut lanes = Vector::<E, NIn>::zero();
        let weight_offs = weight_offs + v * stride_oc;

        for in_c in range_stepped(0, in_c_per_group, vector_size_in as u32) {
            let val = input[(in_offs + in_c as usize) / vector_size_in];

            lanes += val * weight[(weight_offs + in_c as usize) / vector_size_in];
        }

        let mut channel = sum.extract(v);

        #[unroll]
        for i in 0..vector_size_in {
            channel += lanes.extract(i);
        }

        sum.insert(v, channel);
    }
}

/// One input read serves every output channel, which is all a one-step channel loop can win.
#[cube]
fn accumulate_per_step<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    sum: &mut Vector<E, NOut>,
    in_offs: usize,
    weight_offs: usize,
    in_c_per_group: u32,
    stride_oc: usize,
) {
    let vector_size_in = input.vector_size();
    let vector_size_out = sum.vector_size();

    for in_c in range_stepped(0, in_c_per_group, vector_size_in as u32) {
        let in_pos = in_offs + in_c as usize;
        let mut weight_pos = weight_offs + in_c as usize;

        let val = input[in_pos / vector_size_in];

        #[unroll]
        for v in 0..vector_size_out {
            let weight = weight[weight_pos / vector_size_in];
            let val = val * weight;

            #[unroll]
            for i in 0..vector_size_in {
                sum.insert(v, sum.extract(v) + val.extract(i));
            }
            weight_pos += stride_oc;
        }
    }
}

/// The bindings one direct convolution reads and writes.
///
/// The caller owns the output: it allocates it and, where the channel stride is not one, makes
/// the operands contiguous. Both are tensor-library concerns rather than kernel ones.
pub struct DirectTensors {
    pub input: TensorBinding,
    pub weight: TensorBinding,
    pub bias: Option<TensorBinding>,
    pub out: TensorBinding,
}

/// Runs a direct convolution over `N` spatial dimensions.
pub fn launch_direct<const N: usize>(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<N>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    let DirectTensors {
        input,
        weight,
        bias,
        out,
    } = tensors;

    let rank = input.shape.len();
    let dim_c = rank - 1;

    let in_shape = &input.shape[1..dim_c];
    let out_channels = weight.shape[0];
    let kernel_shape = &weight.shape[1..dim_c];
    let out_size = &out.shape[1..dim_c];

    let channels_per_group = out_channels / groups;
    let check_spatial_bounds = should_check_spatial_bounds(in_shape, kernel_shape, out_size, &args);

    // The vector size is taken over `channels_per_group` rather than the whole channel axis, so a
    // grouped convolution never vectorizes across a group boundary.
    let mut grouped_out_shape = out.shape.clone();
    grouped_out_shape[dim_c] = channels_per_group;
    let vector_size_out = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &grouped_out_shape,
        &out.strides,
        dim_c,
    );
    let vector_size_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &weight.shape,
        &weight.strides,
        weight.shape.len() - 1,
    );

    // Only a single-unit plane pays the dependency chain in full; a wide plane hides it and is
    // left with the extra input read per output channel. One lane is exactly as serial as `sum`,
    // and a channel loop of one step has nothing to amortize the fold over.
    let accumulate_lanes = client.properties().hardware.plane_size_max == 1
        && vector_size_in > 1
        && weight.shape[dim_c] > vector_size_in as usize;

    let shape_out = out.shape[1..dim_c].iter().map(|s| *s as u32).collect();
    let shape_out_c = out_channels as u32;

    let mut conv_params = SequenceArg::new();

    for i in 0..kernel_shape.len() {
        conv_params.push(ConvParamLaunch::new(
            args.stride[i] as u32,
            args.dilation[i] as u32,
            args.padding[i] as i32,
        ));
    }

    let working_units = out.shape.iter().product::<usize>() / vector_size_out as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = input
        .required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(out.required_address_type(dtype.size()));

    unsafe {
        direct_conv2d_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size_in,
            vector_size_out,
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|b| b.into_buffer_arg()).into(),
            linear_view(out),
            Conv2dArgsLaunch::new(conv_params, channels_per_group as u32),
            shape_out,
            shape_out_c,
            check_spatial_bounds,
            accumulate_lanes,
            dtype,
        )
    };

    Ok(())
}

fn should_check_spatial_bounds<const N: usize>(
    in_shape: &[usize],
    kernel_shape: &[usize],
    out_shape: &[usize],
    args: &ConvolutionArgs<N>,
) -> bool {
    (0..N).any(|dim| {
        let begin = args.padding[dim] as i64;
        let first = -begin;
        let last = (out_shape[dim] as i64 - 1) * args.stride[dim] as i64
            + (kernel_shape[dim] as i64 - 1) * args.dilation[dim] as i64
            - begin;
        first < 0 || last >= in_shape[dim] as i64
    })
}
