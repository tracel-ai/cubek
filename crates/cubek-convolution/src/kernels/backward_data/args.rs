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
    components::global::memory::{GlobalLayoutConfig, NoopLayout, NoopLayoutLaunch},
    definition::{Blueprint, MatmulElems, MatmulLineSizes, MatrixLayout, TilingBlueprint},
    launch::{
        MatmulArgs, MatmulInputHandleRef, TensorArgs, TensorInputs, TensorInputsLaunch,
        TensorMapArgs, TensorMapInputs, TensorMapInputsLaunch, TensorOutput, TensorOutputLaunch,
    },
    routines::Routine,
};
use enumset::EnumSet;

use crate::components::{
    ConvolutionParams, ConvolutionProblem,
    global::{
        args::{RuntimeArgs, RuntimeArgsLaunch},
        layout::{
            Im2colLayout, Im2colLayoutLaunch, NhwcCheck, NhwcLayout, NhwcLayoutLaunch, OutLayout,
            OutLayoutLaunch, TmaIm2colLayout, TmaIm2colLayoutLaunch, WeightLayout,
            WeightLayoutLaunch,
        },
    },
};

pub trait ConcreteArgs<A: Routine<RuntimeArgs>>:
    MatmulArgs<
        Input<NumericExpand<0>, NumericExpand<1>, NumericExpand<2>>: ConcreteInputsFactory<A>,
        Output<NumericExpand<2>>: ConcreteOutputFactory<A>,
        Config = RuntimeArgs,
    >
{
    fn adjust_problem<R: Runtime>(
        client: &ComputeClient<R>,
        problem: ConvolutionProblem,
        selection: &A::Blueprint,
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
        let padded_channels = problem.out_channels.next_multiple_of(channel_align);
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
        let channel_align = blueprint.tiling_scheme.tile_size.k() as usize;
        let padded_channels = problem.out_channels.next_multiple_of(channel_align);
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
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out_grad: &'a MatmulInputHandleRef<'a, R>,
        weights: &'a MatmulInputHandleRef<'a, R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>);
}

/// Create the output runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory<A: Routine<RuntimeArgs>>: LaunchArg {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: &'a TensorHandleRef<'a, R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, A: Routine<RuntimeArgs>> ConcreteInputsFactory<A>
    for TensorInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out_grad: &'a MatmulInputHandleRef<'a, R>,
        weights: &'a MatmulInputHandleRef<'a, R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        _dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>) {
        type LhsLayout = Chain<NhwcLayout, Im2colLayout>;
        type RhsLayout = Chain<NhwcLayout, WeightLayout>;

        let padded_channels = problem.padded_channels as u32;
        let params = ConvolutionParams::from_problem(problem);

        let layout_nhwc =
            |handle, line_size, checks| NhwcLayoutLaunch::from_handle(handle, line_size, checks);

        let layout_lhs = Im2colLayoutLaunch::from_args(
            client,
            problem,
            params,
            blueprint.lhs_global_layout_config(),
        );
        let layout_rhs =
            WeightLayoutLaunch::from_args(client, problem, blueprint.rhs_global_layout_config());

        let layout_lhs = {
            let mut checks = EnumSet::empty();
            if problem.should_check_spatial_bounds() {
                checks.insert(NhwcCheck::Spatial);
            }
            if problem.should_check_channel() {
                checks.insert(NhwcCheck::Channel);
            }
            let global = layout_nhwc(out_grad.data(), line_sizes.lhs, checks);
            ChainLaunch::new(global, layout_lhs)
        };
        let layout_rhs = {
            let mut checks = EnumSet::empty();
            if problem.should_check_channel() {
                checks.insert(NhwcCheck::Batch);
            }
            let global = layout_nhwc(weights.data(), line_sizes.rhs, checks);
            ChainLaunch::new(global, layout_rhs)
        };

        let inputs = TensorInputsLaunch::new(
            ViewArg::new::<LhsLayout>(out_grad.data().as_array_arg(line_sizes.lhs), layout_lhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new::<RhsLayout>(weights.data().as_array_arg(line_sizes.rhs), layout_rhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            CubeOptionArgs::None,
            CubeOptionArgs::None,
        );

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.k as u32),
            ScalarArg::new(problem.out_channels as u32),
            FastDivmodArgs::<u32>::new(client, padded_channels),
            problem.operation,
        );

        (inputs, runtime_args)
    }
}

impl<EG: Numeric, A: Routine<RuntimeArgs>> ConcreteOutputFactory<A> for TensorOutput<EG> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: &'a TensorHandleRef<'a, R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        type Layout = Chain<NhwcLayout, OutLayout>;

        let global = NhwcLayoutLaunch::from_handle(out, line_sizes.out, EnumSet::empty());
        let layout =
            OutLayoutLaunch::from_args(client, problem, blueprint.out_global_layout_config());
        let layout = ChainLaunch::new(global, layout);
        let view = ViewArg::new::<Layout>(out.as_array_arg(line_sizes.out), layout);
        let batch = VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new());
        TensorOutputLaunch::new(view, batch)
    }
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, A: Routine<RuntimeArgs, Blueprint = TilingBlueprint>>
    ConcreteInputsFactory<A> for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out_grad: &'a MatmulInputHandleRef<'a, R>,
        weights: &'a MatmulInputHandleRef<'a, R>,
        blueprint: &TilingBlueprint,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<'a, R>, RuntimeArgsLaunch<'a, R>) {
        type LhsLayout = TmaIm2colLayout;
        type RhsLayout = WeightLayout;

        let tiling_scheme = blueprint.tiling_scheme;
        let stage_m = tiling_scheme.elements_per_stage_along_m();
        let stage_n = tiling_scheme.elements_per_stage_along_n();
        let stage_k = tiling_scheme.elements_per_stage_along_k();
        let tile_size_k = tiling_scheme.tile_size.k;

        let mut stage_size_rhs = vec![1; problem.dimensionality.num_dims()];
        stage_size_rhs.insert(0, stage_k);
        stage_size_rhs.push(stage_n);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if dtypes.lhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            dtypes.lhs_stage
        };

        let mut elem_stride = vec![1; 2 + problem.stride.len()];

        for (i, stride) in problem.stride.iter().enumerate() {
            elem_stride[i + 1] = *stride as usize;
        }

        let lhs = TensorMapArg::new(
            Im2colArgs {
                pixel_box_lower_corner: calculate_lower_corner(problem),
                pixel_box_upper_corner: calculate_upper_corner(problem),
                channels_per_pixel: tile_size_k,
                pixels_per_column: stage_m,
            },
            out_grad.data().as_tensor_arg(line_sizes.lhs),
            lhs_elem,
        )
        .with_elem_stride(elem_stride);

        let rhs = TensorMapArg::new(
            TiledArgs {
                tile_size: stage_size_rhs,
            },
            weights.data().as_tensor_arg(line_sizes.rhs),
            dtypes.rhs_global,
        );

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
                matrix_layout: MatrixLayout::default(),
            },
        );

        let inputs = TensorMapInputsLaunch::new(
            ViewArg::new_tensor_map_im2col::<LhsLayout, _, _>(lhs, lhs_layout),
            ViewArg::new_tensor_map_tiled::<RhsLayout>(rhs, rhs_layout),
            CubeOptionArgs::None,
            CubeOptionArgs::None,
        );

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(shape_k),
            ScalarArg::new(problem.out_channels as u32),
            FastDivmodArgs::<u32>::new(client, padded_channels),
            problem.operation,
        );

        (inputs, runtime_args)
    }
}

#[allow(clippy::needless_range_loop)]
fn calculate_lower_corner(problem: &ConvolutionProblem) -> Vec<i32> {
    let mut out = vec![0; problem.padding.len()];
    for i in 0..problem.padding.len() {
        out[i] =
            problem.padding[i] - (problem.kernel_size[i] as i32 - 1) * problem.dilation[i] as i32;
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn calculate_upper_corner(problem: &ConvolutionProblem) -> Vec<i32> {
    let mut out = vec![0; problem.padding.len()];
    for i in 0..problem.padding.len() {
        out[i] = problem.padding[i]
            - (problem.kernel_size[i] as i32 - 1) * problem.dilation[i] as i32
            + problem.in_shape[i] as i32
            - problem.out_shape[i] as i32;
    }
    out
}
