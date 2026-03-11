use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::*,
    std::{
        FastDivmodArgs,
        tensor::{
            launch::ViewArg,
            layout::{
                VirtualLayoutLaunch,
                chain::{Chain, ChainLaunch},
            },
        },
    },
    zspace::{shape, strides},
};
use cubek_matmul::{
    components::global::memory::{GlobalLayoutConfig, NoopLayout, NoopLayoutLaunch},
    definition::{Blueprint, MatmulElems, MatmulVectorSizes, TilingBlueprint},
    launch::{
        MatmulArgs, MatmulInputBinding, TensorArgs, TensorInputs, TensorInputsLaunch,
        TensorMapArgs, TensorMapInputs, TensorMapInputsLaunch, TensorOutput, TensorOutputLaunch,
    },
    routines::Routine,
};
use cubek_std::{MatrixLayout, stage::SwizzleMode};
use enumset::EnumSet;

use crate::components::{
    ConvolutionParams, ConvolutionProblem,
    global::{
        args::{RuntimeArgs, RuntimeArgsLaunch},
        layout::{
            BiasLayout, BiasLayoutLaunch, Im2colLayout, Im2colLayoutLaunch, NhwcCheck, NhwcLayout,
            NhwcLayoutLaunch, OutLayout, OutLayoutLaunch, TmaIm2colLayout, TmaIm2colLayoutLaunch,
            WeightLayout, WeightLayoutLaunch,
        },
    },
};

pub trait ConcreteArgs<A: Routine<RuntimeArgs>>:
    MatmulArgs<
        Input<Vector<NumericExpand<0>, SizeExpand<1>>, Vector<NumericExpand<2>, SizeExpand<3>>, Vector<NumericExpand<4>, SizeExpand<5>>>: ConcreteInputsFactory<A>,
        Output<Vector<NumericExpand<4>, SizeExpand<5>>>: ConcreteOutputFactory<A>,
        Config = RuntimeArgs,
    >
{
    fn adjust_problem<R: Runtime>(
        client: &ComputeClient<R>,
        problem: ConvolutionProblem,
        blueprint: &A::Blueprint,
        dtypes: &MatmulElems,
    ) -> ConvolutionProblem;
}

impl<A: Routine<RuntimeArgs>> ConcreteArgs<A> for TensorArgs<RuntimeArgs> {
    fn adjust_problem<R: Runtime>(
        client: &ComputeClient<R>,
        mut problem: ConvolutionProblem,
        _blueprint: &A::Blueprint,
        dtypes: &MatmulElems,
    ) -> ConvolutionProblem {
        let load_width = client.properties().hardware.load_width;
        let channel_align = load_width as usize / dtypes.lhs_global.size_bits();
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_k = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.k = shape_k;
        problem.padded_channels = padded_channels;

        problem
    }
}

impl<A: Routine<RuntimeArgs, Blueprint = TilingBlueprint>> ConcreteArgs<A>
    for TensorMapArgs<RuntimeArgs>
{
    fn adjust_problem<R: Runtime>(
        _client: &ComputeClient<R>,
        mut problem: ConvolutionProblem,
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
    ) -> ConvolutionProblem {
        let channel_align = match blueprint.swizzle_modes.lhs {
            SwizzleMode::None => blueprint.tiling_scheme.tile_size.k() as usize,
            _ => blueprint.tiling_scheme.elements_per_stage_along_k() as usize,
        };
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_k = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.k = shape_k;
        problem.padded_channels = padded_channels;

        problem
    }
}

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory<A: Routine<RuntimeArgs>>: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<R: Runtime>(
        client: &ComputeClient<R>,
        lhs: MatmulInputBinding<R>,
        rhs: MatmulInputBinding<R>,
        bias: Option<MatmulInputBinding<R>>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        vector_sizes: &MatmulVectorSizes,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<R>, RuntimeArgsLaunch<R>);
}

/// Create the output runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory<A: Routine<RuntimeArgs>>: LaunchArg {
    fn create<R: Runtime>(
        client: &ComputeClient<R>,
        out: TensorBinding<R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        vector_sizes: &MatmulVectorSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<R>;
}

impl<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive, A: Routine<RuntimeArgs>>
    ConcreteInputsFactory<A> for TensorInputs<Lhs, Rhs, EO>
{
    fn create<R: Runtime>(
        client: &ComputeClient<R>,
        lhs: MatmulInputBinding<R>,
        rhs: MatmulInputBinding<R>,
        bias: Option<MatmulInputBinding<R>>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        vector_sizes: &MatmulVectorSizes,
        _dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<R>, RuntimeArgsLaunch<R>) {
        type LhsLayout = Chain<NhwcLayout, Im2colLayout>;
        type RhsLayout = Chain<NhwcLayout, WeightLayout>;

        let padded_channels = problem.padded_channels as u32;
        let conv_params = ConvolutionParams::from_problem(problem);

        let layout_nhwc = |handle, vector_size, checks| {
            NhwcLayoutLaunch::from_handle(handle, vector_size, checks)
        };
        let layout_lhs = Im2colLayoutLaunch::from_args(
            client,
            problem,
            conv_params,
            blueprint.lhs_global_layout_config(),
        );
        let layout_rhs =
            WeightLayoutLaunch::from_args(client, problem, blueprint.rhs_global_layout_config());
        let layout_bias = BiasLayoutLaunch::new(problem.n as u32, vector_sizes.out as u32);

        let layout_lhs = {
            let mut checks = EnumSet::empty();
            if problem.should_check_spatial_bounds() {
                checks.insert(NhwcCheck::Spatial);
            }
            if problem.should_check_channel() {
                checks.insert(NhwcCheck::Channel);
            }
            let global = layout_nhwc(lhs.data(), vector_sizes.lhs, checks);
            ChainLaunch::new(global, layout_lhs)
        };
        let layout_rhs = {
            let mut checks = EnumSet::empty();
            if problem.should_check_channel() {
                checks.insert(NhwcCheck::Channel);
            }
            let global = layout_nhwc(rhs.data(), vector_sizes.rhs, checks);
            ChainLaunch::new(global, layout_rhs)
        };

        let inputs = TensorInputsLaunch::new(
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new::<LhsLayout>(lhs.into_data().into_array_arg(), layout_lhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new::<RhsLayout>(rhs.into_data().into_array_arg(), layout_rhs),
            bias.as_ref()
                .map(|_| VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()))
                .into(),
            bias.map(|bias| {
                ViewArg::new::<BiasLayout>(bias.into_data().into_array_arg(), layout_bias)
            })
            .into(),
        );

        let runtime_args = RuntimeArgsLaunch::new(
            problem.k as u32,
            problem.channels as u32,
            FastDivmodArgs::<u32>::new(client, padded_channels),
            conv_params.operation,
        );

        (inputs, runtime_args)
    }
}

impl<EG: CubePrimitive, A: Routine<RuntimeArgs>> ConcreteOutputFactory<A> for TensorOutput<EG> {
    fn create<R: Runtime>(
        client: &ComputeClient<R>,
        out: TensorBinding<R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        vector_sizes: &MatmulVectorSizes,
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<R> {
        type Layout = Chain<NhwcLayout, OutLayout>;

        let global = NhwcLayoutLaunch::from_handle(&out, vector_sizes.out, EnumSet::empty());
        let layout =
            OutLayoutLaunch::from_args(client, problem, blueprint.out_global_layout_config());
        let layout = ChainLaunch::new(global, layout);
        let view = ViewArg::new::<Layout>(out.into_array_arg(), layout);
        let batch = VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new());
        TensorOutputLaunch::new(view, batch)
    }
}

impl<
    Lhs: CubePrimitive,
    Rhs: CubePrimitive,
    EO: CubePrimitive,
    A: Routine<RuntimeArgs, Blueprint = TilingBlueprint>,
> ConcreteInputsFactory<A> for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<R: Runtime>(
        client: &ComputeClient<R>,
        lhs: MatmulInputBinding<R>,
        rhs: MatmulInputBinding<R>,
        bias: Option<MatmulInputBinding<R>>,
        blueprint: &TilingBlueprint,
        problem: &ConvolutionProblem,
        vector_sizes: &MatmulVectorSizes,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<R>, RuntimeArgsLaunch<R>) {
        let tiling_scheme = blueprint.tiling_scheme;
        let stage_m = tiling_scheme.elements_per_stage_along_m();
        let stage_n = tiling_scheme.elements_per_stage_along_n();

        let tile_size_k = match blueprint.swizzle_modes.lhs {
            SwizzleMode::None => tiling_scheme.tile_size.k,
            _ => tiling_scheme.elements_per_stage_along_k(),
        };

        let mut stage_size_rhs = shape![1; problem.dimensionality.num_dims()];
        stage_size_rhs.insert(0, stage_n as usize);
        stage_size_rhs.push(tile_size_k as usize);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if dtypes.lhs_stage == f32::as_type_native_unchecked().storage_type() {
            tf32::as_type_native_unchecked().storage_type()
        } else {
            dtypes.lhs_stage
        };

        let mut elem_stride = strides![1; 2 + problem.stride.len()];

        for (i, stride) in problem.stride.iter().enumerate() {
            elem_stride[i + 1] = *stride as usize;
        }

        let lhs = TensorMapArg::new(
            Im2colArgs {
                pixel_box_lower_corner: calculate_lower_corner(&problem.padding),
                pixel_box_upper_corner: calculate_upper_corner(
                    &problem.padding,
                    &problem.kernel_size,
                    &problem.dilation,
                ),
                channels_per_pixel: tile_size_k,
                pixels_per_column: stage_m,
            },
            lhs.clone().into_data().into_tensor_arg(),
            lhs_elem,
        )
        .with_elem_stride(elem_stride)
        .with_swizzle(blueprint.swizzle_modes.lhs.into());

        let rhs = TensorMapArg::new(
            TiledArgs {
                tile_size: stage_size_rhs,
            },
            rhs.clone().into_data().into_tensor_arg(),
            dtypes.rhs_global,
        )
        .with_swizzle(blueprint.swizzle_modes.rhs.into());

        let padded_channels = problem.padded_channels as u32;
        let shape_k = problem.k as u32;

        // Im2col needs extra checking because if `k` is OOB it wraps around the kernel and can load
        // in-bounds but not in-kernel elements. Other TMA layouts are always outside the shape if
        // any matrix dim is out of bounds.
        let stages_lhs = A::num_stages().lhs;
        let stages_size_k = blueprint.tiling_scheme.elements_per_stage_along_k() * stages_lhs;
        let check_kernel = !shape_k.is_multiple_of(stages_size_k);
        let lhs_layout = TmaIm2colLayoutLaunch::from_args(client, problem, check_kernel);
        let rhs_layout = WeightLayoutLaunch::from_args(
            client,
            problem,
            GlobalLayoutConfig {
                check_row_bounds: false,
                check_col_bounds: false,
                matrix_layout: MatrixLayout::ColMajor,
            },
        );

        let bias = bias.map(|bias| {
            let layout = BiasLayoutLaunch::new(problem.n as u32, vector_sizes.out as u32);
            ViewArg::new::<BiasLayout>(bias.into_data().into_array_arg(), layout)
        });

        let inputs = TensorMapInputsLaunch::new(
            ViewArg::new_tensor_map_im2col::<TmaIm2colLayout, _, _>(lhs, lhs_layout),
            ViewArg::new_tensor_map_tiled::<WeightLayout>(rhs, rhs_layout),
            bias.into(),
            ComptimeOptionArgs::Some(VirtualLayoutLaunch::new::<NoopLayout>(
                NoopLayoutLaunch::new(),
            )),
        );

        let runtime_args = RuntimeArgsLaunch::new(
            shape_k,
            problem.channels as u32,
            FastDivmodArgs::<u32>::new(client, padded_channels),
            problem.operation,
        );

        (inputs, runtime_args)
    }
}

fn calculate_lower_corner(padding: &[i32]) -> Vec<i32> {
    padding.iter().map(|padding| -*padding).collect()
}

fn calculate_upper_corner(padding: &[i32], kernel_size: &[u32], dilation: &[u32]) -> Vec<i32> {
    padding
        .iter()
        .zip(kernel_size)
        .zip(dilation)
        .map(|((padding, kernel_size), dilation)| {
            *padding - (*kernel_size - 1) as i32 * *dilation as i32
        })
        .collect()
}
