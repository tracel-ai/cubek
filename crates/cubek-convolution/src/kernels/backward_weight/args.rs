use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::*,
    server::TensorMapMeta,
    std::{
        CubeOptionArgs, FastDivmodArgs,
        tensor::{
            launch::ViewArg,
            layout::{
                VirtualLayoutLaunch,
                chain::{Chain, ChainLaunch},
            },
        },
    },
};
use cubek_matmul::{
    components::{
        global::{
            GlobalConfig as _,
            memory::{NoopLayout, NoopLayoutLaunch, Transpose, TransposeLaunch},
        },
        stage::StageConfig as _,
    },
    definition::{MatmulElems, MatmulLineSizes, TilingBlueprint},
    launch::{
        MatmulArgs, MatmulInputHandleRef, TensorArgs, TensorInputs, TensorInputsLaunch,
        TensorMapArgs, TensorMapInputs, TensorMapInputsLaunch, TensorOutput, TensorOutputLaunch,
    },
};
use enumset::EnumSet;

use crate::components::{
    ConvGemmConfig, ConvolutionParams, ConvolutionProblem,
    global::{
        args::RuntimeArgsLaunch,
        layout::{
            Im2colLayout, Im2colLayoutLaunch, NhwcCheck, NhwcLayout, NhwcLayoutLaunch, OutLayout,
            OutLayoutLaunch, TmaIm2colLayout, TmaIm2colLayoutLaunch, TmaOutGradLayout,
            TmaOutGradLayoutLaunch, WeightLayout, WeightLayoutLaunch,
        },
    },
};

pub trait ConcreteArgs:
    MatmulArgs<
        Input<NumericExpand<0>, NumericExpand<1>, NumericExpand<2>>: ConcreteInputsFactory,
        Output<NumericExpand<2>>: ConcreteOutputFactory,
    >
{
    fn adjust_problem<R: Runtime>(
        client: &ComputeClient<R>,
        problem: ConvolutionProblem,
        selection: &TilingBlueprint,
        dtypes: &MatmulElems,
    ) -> ConvolutionProblem;
}

impl ConcreteArgs for TensorArgs {
    fn adjust_problem<R: Runtime>(
        client: &ComputeClient<R>,
        mut problem: ConvolutionProblem,
        _selection: &TilingBlueprint,
        dtypes: &MatmulElems,
    ) -> ConvolutionProblem {
        let load_width = client.properties().hardware.load_width;
        let channel_align = load_width as usize / dtypes.lhs_global.size_bits();
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_n = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.n = shape_n;
        problem.padded_channels = padded_channels;

        problem
    }
}

impl ConcreteArgs for TensorMapArgs {
    fn adjust_problem<R: Runtime>(
        _client: &ComputeClient<R>,
        mut problem: ConvolutionProblem,
        selection: &TilingBlueprint,
        _dtypes: &MatmulElems,
    ) -> ConvolutionProblem {
        let channel_align = selection.tiling_scheme.tile_size.n() as usize;
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_n = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.n = shape_n;
        problem.padded_channels = padded_channels;

        problem
    }
}

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        input: &'a MatmulInputHandleRef<'a, R>,
        out_grad: &'a MatmulInputHandleRef<'a, R>,
        selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>);
}

/// Create the output runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: &'a TensorHandleRef<'a, R>,
        selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory for TensorInputs<Lhs, Rhs, EO> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        input: &'a MatmulInputHandleRef<'a, R>,
        out_grad: &'a MatmulInputHandleRef<'a, R>,
        _selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        _dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>) {
        type LhsLayout = Chain<NhwcLayout, Transpose<OutLayout>>;
        type RhsLayout = Chain<NhwcLayout, Im2colLayout>;

        let padded_channels = problem.padded_channels as u32;

        let layout_nhwc = |handle, line_size, checks| {
            NhwcLayoutLaunch::from_handle(handle, line_size as u32, checks)
        };

        let layout_lhs =
            OutLayoutLaunch::from_args(client, problem, config.lhs_global_memory_config());
        let layout_rhs = Im2colLayoutLaunch::from_args(
            client,
            problem,
            config.params(),
            config.rhs_global_memory_config(),
        );

        let layout_lhs = {
            let global = layout_nhwc(out_grad.data(), line_sizes.lhs, EnumSet::empty());
            ChainLaunch::new(global, TransposeLaunch::new(layout_lhs))
        };
        let layout_rhs = {
            let mut checks = EnumSet::empty();
            if problem.should_check_spatial_bounds() {
                checks.insert(NhwcCheck::Spatial);
            }
            if problem.should_check_channel() {
                checks.insert(NhwcCheck::Channel);
            }
            let global = layout_nhwc(input.data(), line_sizes.rhs, checks);
            ChainLaunch::new(global, layout_rhs)
        };

        let inputs = TensorInputsLaunch::new(
            ViewArg::new::<LhsLayout>(out_grad.data().as_array_arg(line_sizes.lhs), layout_lhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new::<RhsLayout>(input.data().as_array_arg(line_sizes.rhs), layout_rhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            CubeOptionArgs::None,
            CubeOptionArgs::None,
        );

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.k as u32),
            ScalarArg::new(problem.channels as u32),
            FastDivmodArgs::new(client, padded_channels),
            config.operation(),
        );

        (inputs, runtime_args)
    }
}

impl<EG: Numeric> ConcreteOutputFactory for TensorOutput<EG> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: &'a TensorHandleRef<'a, R>,
        _selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R> {
        // Weight layout assumes col-major so it's technically "transposed" when it's row-major.
        // Should look into maybe inverting this and using `Transpose` for forward instead.
        type Layout = Chain<NhwcLayout, Transpose<WeightLayout>>;

        let mut checks = EnumSet::empty();
        if problem.should_check_channel() {
            checks.insert(NhwcCheck::Channel);
        }
        let global = NhwcLayoutLaunch::from_handle(out, line_sizes.out as u32, checks);
        let layout =
            WeightLayoutLaunch::from_args(client, problem, config.out_global_memory_config());
        let layout = ChainLaunch::new(global, TransposeLaunch::new(layout));
        let view = ViewArg::new::<Layout>(out.as_array_arg(line_sizes.out), layout);
        let batch = VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new());
        TensorOutputLaunch::new(view, batch)
    }
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory
    for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        input: &'a MatmulInputHandleRef<'a, R>,
        out_grad: &'a MatmulInputHandleRef<'a, R>,
        selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>) {
        type LhsLayout = Transpose<TmaOutGradLayout>;
        type RhsLayout = TmaIm2colLayout;

        let tiling_scheme = selection.tiling_scheme;
        let stage_k = tiling_scheme.elements_per_stage_along_k();
        let tile_size_m = tiling_scheme.tile_size.m;
        let tile_size_n = tiling_scheme.tile_size.n;

        let dim_c = out_grad.shape().len() - 1;
        let stage_size_lhs = vec![stage_k, tile_size_m];

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if *dtypes.lhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            *dtypes.lhs_stage
        };
        let rhs_elem = if *dtypes.rhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            *dtypes.rhs_stage
        };

        let mut elem_stride = vec![1; 2 + problem.stride.len()];

        for (i, stride) in problem.stride.iter().enumerate() {
            elem_stride[i + 1] = *stride as usize;
        }

        let lhs_shape = vec![problem.k, problem.m];
        let lhs_strides = vec![
            out_grad.data().strides[dim_c - 1],
            out_grad.data().strides[dim_c],
        ];

        let lhs_meta = TensorMapMeta {
            format: TensorMapFormat::Tiled(TiledArgs {
                tile_size: stage_size_lhs,
            }),
            rank: 2,
            shape: lhs_shape,
            strides: lhs_strides,
            elem_stride: vec![1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: TensorMapPrefetch::None,
            oob_fill: OobFill::Zero,
            storage_ty: lhs_elem,
        };

        let lhs = TensorMapArg {
            tensor: out_grad.data().as_tensor_arg(line_sizes.lhs),
            metadata: lhs_meta,
            _kind: core::marker::PhantomData,
        };

        let rhs = TensorMapArg::new(
            Im2colArgs {
                pixel_box_lower_corner: calculate_lower_corner(&problem.padding),
                pixel_box_upper_corner: calculate_upper_corner(
                    &problem.padding,
                    &problem.kernel_size,
                    &problem.dilation,
                ),
                channels_per_pixel: tile_size_n,
                pixels_per_column: stage_k,
            },
            input.data().as_tensor_arg(line_sizes.rhs),
            rhs_elem,
        )
        .with_elem_stride(elem_stride);

        let padded_channels = problem.padded_channels as u32;
        let shape_k = problem.k as u32;
        let shape_n = problem.n as u32;

        let shape_out = problem
            .out_shape
            .iter()
            .map(|it| FastDivmodArgs::new(client, *it as u32))
            .collect();

        // Im2col needs extra checking because if `n` is OOB it wraps around the kernel and can load
        // in-bounds but not in-kernel elements. Other TMA layouts are always outside the shape if
        // any matrix dim is out of bounds.
        let stages_rhs = config.stage_config().rhs_smem_config().num_stages;
        let stages_size_n = selection.tiling_scheme.elements_per_stage_along_n() * stages_rhs;

        let lhs_layout = TmaOutGradLayoutLaunch::new();
        let lhs_layout = TransposeLaunch::new(lhs_layout);

        let rhs_layout = TmaIm2colLayoutLaunch::new(
            shape_out,
            FastDivmodArgs::new(client, padded_channels),
            ConvolutionParams::from_problem(problem),
            !shape_n.is_multiple_of(stages_size_n),
        );

        let inputs = TensorMapInputsLaunch::new(
            ViewArg::new_tensor_map_tiled::<LhsLayout>(lhs, lhs_layout),
            ViewArg::new_tensor_map_im2col::<RhsLayout, _, _>(rhs, rhs_layout),
            CubeOptionArgs::None,
            CubeOptionArgs::None,
        );

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(shape_k),
            ScalarArg::new(problem.channels as u32),
            FastDivmodArgs::new(client, padded_channels),
            config.operation(),
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
