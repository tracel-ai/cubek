use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::*,
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
            memory::{NoopLayout, NoopLayoutLaunch},
        },
        stage::StageConfig as _,
    },
    definition::{MatmulElems, MatmulLineSizes, TilingBlueprint},
    launch::{
        MatmulArgs, MatmulInputHandleRef, TensorArgs, TensorInputs, TensorInputsLaunch,
        TensorMapArgs, TensorMapInputs, TensorMapInputsLaunch, TensorOutput, TensorOutputLaunch,
    },
};

use crate::components::{
    ConvGemmConfig, ConvolutionParams, ConvolutionProblem,
    global::{
        args::RuntimeArgsLaunch,
        layout::{
            BiasLayout, BiasLayoutLaunch, Im2colLayout, Im2colLayoutLaunch, NhwcLayout,
            NhwcLayoutLaunch, OutLayout, OutLayoutLaunch, TmaIm2colLayout, TmaIm2colLayoutLaunch,
            WeightLayout, WeightLayoutLaunch,
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
        let shape_k = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.k = shape_k;
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
        let channel_align = selection.tiling_scheme.tile_size.k() as usize;
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_k = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.k = shape_k;
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
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a MatmulInputHandleRef<'a, R>>,
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
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory for TensorInputs<Lhs, Rhs, EO> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a MatmulInputHandleRef<'a, R>>,
        _selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        _dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>) {
        type LhsLayout = Chain<NhwcLayout, Im2colLayout>;
        type RhsLayout = Chain<NhwcLayout, WeightLayout>;

        let padded_channels = problem.padded_channels as u32;

        let layout_nhwc = |handle, line_size, check_spatial| {
            NhwcLayoutLaunch::from_handle(
                handle,
                line_size as u32,
                check_spatial,
                problem.check_channel(),
            )
        };
        let layout_lhs = Im2colLayoutLaunch::from_args(
            client,
            problem,
            config.params(),
            config.lhs_global_memory_config(),
        );
        let layout_rhs =
            WeightLayoutLaunch::from_args(client, problem, config.rhs_global_memory_config());
        let layout_bias =
            BiasLayoutLaunch::new(ScalarArg::new(problem.n as u32), line_sizes.out as u32);

        let layout_lhs = {
            let global = layout_nhwc(lhs.data(), line_sizes.lhs, config.check_spatial_bounds());
            ChainLaunch::new(global, layout_lhs)
        };
        let layout_rhs = {
            let global = layout_nhwc(rhs.data(), line_sizes.rhs, false);
            ChainLaunch::new(global, layout_rhs)
        };

        let inputs = TensorInputsLaunch::new(
            ViewArg::new::<LhsLayout>(lhs.data().as_array_arg(line_sizes.lhs), layout_lhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new::<RhsLayout>(rhs.data().as_array_arg(line_sizes.rhs), layout_rhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            bias.map(|bias| {
                ViewArg::new::<BiasLayout>(bias.data().as_array_arg(line_sizes.out), layout_bias)
            })
            .into(),
            bias.map(|_| VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()))
                .into(),
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
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        type Layout = Chain<NhwcLayout, OutLayout>;

        let global = NhwcLayoutLaunch::from_handle(out, line_sizes.out as u32, false, false);
        let layout = OutLayoutLaunch::from_args(client, problem, config.out_global_memory_config());
        let layout = ChainLaunch::new(global, layout);
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
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a MatmulInputHandleRef<'a, R>>,
        selection: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>) {
        let tiling_scheme = selection.tiling_scheme;
        let stage_m = tiling_scheme.elements_per_stage_along_m();
        let stage_n = tiling_scheme.elements_per_stage_along_n();
        let tile_size_k = tiling_scheme.tile_size.k;

        let mut stage_size_rhs = vec![1; problem.dimensionality.num_dims() as usize];
        stage_size_rhs.insert(0, stage_n);
        stage_size_rhs.push(tile_size_k);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if *dtypes.lhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            *dtypes.lhs_stage
        };

        let mut elem_stride = vec![1; 2 + problem.stride.len()];

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
            lhs.data().as_tensor_arg(line_sizes.lhs),
            lhs_elem,
        )
        .with_elem_stride(elem_stride);

        let rhs = TensorMapArg::new(
            TiledArgs {
                tile_size: stage_size_rhs,
            },
            rhs.data().as_tensor_arg(1),
            *dtypes.rhs_global,
        );

        let padded_channels = problem.padded_channels as u32;
        let shape_k = problem.k as u32;

        let shape_out = problem
            .out_shape
            .iter()
            .map(|it| FastDivmodArgs::new(client, *it as u32))
            .collect();

        // Im2col needs extra checking because if `k` is OOB it wraps around the kernel and can load
        // in-bounds but not in-kernel elements. Other TMA layouts are always outside the shape if
        // any matrix dim is out of bounds.
        let stages_lhs = config.stage_config().lhs_smem_config().num_stages;
        let stages_size_k = selection.tiling_scheme.elements_per_stage_along_k() * stages_lhs;
        let lhs_layout = TmaIm2colLayoutLaunch::new(
            shape_out,
            FastDivmodArgs::new(client, padded_channels),
            ConvolutionParams::from_problem(problem),
            !shape_k.is_multiple_of(stages_size_k),
        );
        let rhs_layout = WeightLayoutLaunch::from_args(client, problem, Default::default());

        let bias = bias.map(|bias| {
            let layout =
                BiasLayoutLaunch::new(ScalarArg::new(problem.n as u32), line_sizes.out as u32);
            ViewArg::new::<BiasLayout>(bias.data().as_array_arg(line_sizes.out), layout)
        });

        let inputs = TensorMapInputsLaunch::new(
            ViewArg::new_tensor_map_im2col::<TmaIm2colLayout, _, _>(lhs, lhs_layout),
            ViewArg::new_tensor_map_tiled::<WeightLayout>(rhs, rhs_layout),
            bias.into(),
            CubeOptionArgs::Some(VirtualLayoutLaunch::new::<NoopLayout>(
                NoopLayoutLaunch::new(),
            )),
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
