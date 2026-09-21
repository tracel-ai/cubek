use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    ir::{FloatKind, VectorRegisters},
    num_traits::Zero,
    prelude::*,
    std::tensor::layout::linear::{LinearViewMut, linear_view},
    std::{FastDivmod, FastDivmodInt},
    tensor_vector_size_parallel,
};

use crate::{components::ConvSetupError, launch::ConvolutionArgs};

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
    #[comptime] channel_block: usize,
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

    let vector_size_in = input.vector_size();

    if accumulate_lanes {
        #[unroll]
        for bi in 0..comptime![vector_size_out / channel_block] {
            let base_v = bi * channel_block;
            let mut lanes = Array::<Vector<E, NIn>>::new(channel_block);

            kernel_loop(
                input,
                weight,
                &mut sum,
                &mut lanes,
                in_offs,
                true,
                weight_offs,
                &loop_params,
                0usize,
                has_padding,
                accumulate_lanes,
                base_v,
            );

            #[unroll]
            for j in 0..channel_block {
                let mut channel = sum.extract(base_v + j);

                #[unroll]
                for i in 0..vector_size_in {
                    channel += lanes[j].extract(i);
                }

                sum.insert(base_v + j, channel);
            }
        }
    } else {
        // Unread: `accumulate_per_step` sums into `sum`, and the loop still takes an accumulator.
        let mut lanes = Array::<Vector<E, NIn>>::new(1usize);

        kernel_loop(
            input,
            weight,
            &mut sum,
            &mut lanes,
            in_offs,
            true,
            weight_offs,
            &loop_params,
            0usize,
            has_padding,
            accumulate_lanes,
            0usize,
        );
    }

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
    lanes: &mut Array<Vector<E, NIn>>,
    in_offs: usize,
    in_bounds: bool,
    weight_offs: usize,
    params: &LoopParams,
    #[comptime] kernel_dim: usize,
    #[comptime] has_padding: bool,
    #[comptime] accumulate_lanes: bool,
    #[comptime] base_v: usize,
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
                lanes,
                in_offs,
                in_bounds,
                weight_offs,
                params,
                comptime![kernel_dim + 1],
                has_padding,
                accumulate_lanes,
                base_v,
            );
        }
    } else {
        kernel_loop_inner(
            input,
            weight,
            sum,
            lanes,
            in_offs,
            in_bounds,
            weight_offs,
            params.in_c_per_group,
            params.stride_oc,
            accumulate_lanes,
            base_v,
        );
    }
}

#[cube]
fn kernel_loop_inner<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    sum: &mut Vector<E, NOut>,
    lanes: &mut Array<Vector<E, NIn>>,
    in_offs: usize,
    in_bounds: bool,
    weight_offs: usize,
    in_c_per_group: u32,
    stride_oc: usize,
    #[comptime] accumulate_lanes: bool,
    #[comptime] base_v: usize,
) {
    if in_bounds {
        if accumulate_lanes {
            accumulate_in_lanes(
                input,
                weight,
                lanes,
                in_offs,
                weight_offs,
                in_c_per_group,
                stride_oc,
                base_v,
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

/// One input read serves a whole block of output channels, and the block's accumulator outlives
/// the kernel window, so a channel step costs one input load per block and never a fold.
#[cube]
fn accumulate_in_lanes<E: Numeric, NIn: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    lanes: &mut Array<Vector<E, NIn>>,
    in_offs: usize,
    weight_offs: usize,
    in_c_per_group: u32,
    stride_oc: usize,
    #[comptime] base_v: usize,
) {
    let vector_size_in = input.vector_size();
    let block = lanes.len();

    for in_c in range_stepped(0, in_c_per_group, vector_size_in as u32) {
        let val = input[(in_offs + in_c as usize) / vector_size_in];

        #[unroll]
        for j in 0..block {
            let weight_offs = weight_offs + (base_v + j) * stride_oc + in_c as usize;

            lanes[j] += val * weight[weight_offs / vector_size_in];
        }
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

    let hardware = &client.properties().hardware;
    let io_vector_sizes = client.io_optimized_vector_sizes(dtype.size());

    // Use channels_per_group instead of in_channels to avoid issues here
    let vector_size_in = tensor_vector_size_parallel(
        io_vector_sizes.clone(),
        &weight.shape,
        &weight.strides,
        weight.shape.len() - 1,
    );

    // Only a single-unit plane pays the dependency chain in full; a wide plane hides it and is
    // left with the extra input read per output channel. One lane is exactly as serial as `sum`,
    // and a channel loop of one step has nothing to amortize the fold over.
    let blocks = VectorRegisters::of(hardware, lane_size(dtype))
        .filter(|_| {
            hardware.plane_size_max == 1
                && vector_size_in > 1
                && weight.shape[dim_c] > vector_size_in as usize
        })
        .map(|registers| ChannelBlocks::new(registers, vector_size_in));

    // Need custom vector size calculation here to account for the groups division. Need to vectorize
    // over `channels_per_group` instead.
    let mut grouped_out_shape = out.shape.clone();
    grouped_out_shape[dim_c] = channels_per_group;
    let widest_out = blocks.map_or_else(
        || io_vector_sizes.max().unwrap_or(1),
        |blocks| blocks.output_lanes,
    );
    let vector_size_out = tensor_vector_size_parallel(
        (0..=widest_out.ilog2()).map(|power| 1 << power),
        &grouped_out_shape,
        &out.strides,
        dim_c,
    );

    let accumulate_lanes = blocks.is_some();
    let channel_block = blocks.map_or(1, |blocks| blocks.block.min(vector_size_out));

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
            channel_block,
            dtype,
        )
    };

    Ok(())
}

/// How many output channels share one input read, and how many a unit spans.
#[derive(Debug, Clone, Copy)]
struct ChannelBlocks {
    /// A power of two, so it divides any output vector at least as wide.
    block: usize,
    output_lanes: usize,
}

impl ChannelBlocks {
    /// The block takes half the registers, so the input read, the weight loads and the products
    /// keep the rest. An accumulator that does not fit spills, and every add becomes a load and a
    /// store.
    fn new(registers: VectorRegisters, vector_size_in: usize) -> Self {
        let block = registers
            .vectors_fitting(vector_size_in, registers.count() / 2)
            .max(1);
        let block = 1 << block.ilog2();

        // The output vector outlives every block, so it gets one accumulator's share of the
        // registers, and a unit pays its dispatch, position and bias once for several blocks.
        Self {
            block,
            output_lanes: registers.widest_lanes(block),
        }
    }
}

/// A host without half arithmetic evaluates a half float in f32 registers.
fn lane_size(dtype: ElemType) -> usize {
    match dtype {
        ElemType::Float(FloatKind::F16 | FloatKind::BF16) => size_of::<f32>(),
        dtype => dtype.size(),
    }
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

#[cfg(test)]
mod tests {
    use cubecl::ir::HardwareProperties;

    use super::*;

    fn registers(load_width: u32, count: u32, elem_size: usize) -> VectorRegisters {
        let hardware = HardwareProperties {
            load_width,
            vector_register_count: Some(count),
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size: 48 * 1024,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: 16,
            max_cube_dim: (16, 16, 16),
            num_streaming_multiprocessors: None,
            num_cpu_cores: Some(16),
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: usize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };
        VectorRegisters::of(&hardware, elem_size).unwrap()
    }

    fn avx2(elem_size: usize) -> VectorRegisters {
        registers(256, 16, elem_size)
    }

    fn blocks(registers: VectorRegisters, vector_size_in: usize) -> (usize, usize) {
        let blocks = ChannelBlocks::new(registers, vector_size_in);
        (blocks.block, blocks.output_lanes)
    }

    #[test]
    fn a_two_register_accumulator_halves_the_block() {
        assert_eq!(blocks(avx2(4), 16).0, 4);
    }

    #[test]
    fn a_half_float_accumulator_spends_f32_registers() {
        let f16 = avx2(lane_size(ElemType::Float(FloatKind::F16)));
        assert_eq!(blocks(f16, 16), (4, 32));
    }

    #[test]
    fn a_register_wide_accumulator_spends_one() {
        assert_eq!(blocks(avx2(4), 8).0, 8);
        assert_eq!(blocks(registers(512, 32, 4), 16).0, 16);
    }

    #[test]
    fn a_narrow_accumulator_still_spends_a_whole_register() {
        assert_eq!(blocks(avx2(4), 2).0, 8);
    }

    #[test]
    fn an_accumulator_wider_than_the_budget_still_gets_a_block_of_one() {
        assert_eq!(blocks(avx2(8), 64).0, 1);
    }

    #[test]
    fn a_unit_spans_two_blocks_of_register_wide_accumulators() {
        assert_eq!(blocks(avx2(4), 8), (8, 16));
        assert_eq!(blocks(registers(512, 32, 4), 16), (16, 32));
    }
}
