use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::*,
    server::TensorMapMeta,
    std::tensor::{
        launch::ViewArg,
        layout::{
            VirtualLayoutLaunch,
            chain::{Chain, ChainLaunch},
        },
    },
    zspace::{metadata::Metadata, shape, strides},
};
use cubek_matmul::{
    components::global::memory::{NoopLayout, NoopLayoutLaunch, Transpose, TransposeLaunch},
    definition::{Blueprint, MatmulElems, TilingBlueprint},
    launch::*,
    routines::Routine,
};
use cubek_std::stage::SwizzleMode;
use enumset::EnumSet;

use crate::components::{
    ConvolutionParams, ConvolutionProblem,
    global::{
        args::{RuntimeArgs, RuntimeArgsLaunch},
        layout::{
            Im2colLayout, Im2colLayoutLaunch, NhwcCheck, NhwcLayout, NhwcLayoutLaunch, OutLayout,
            OutLayoutLaunch, TmaIm2colLayout, TmaIm2colLayoutLaunch, TmaOutGradLayout,
            TmaOutGradLayoutLaunch, WeightLayout, WeightLayoutLaunch,
        },
    },
};

pub trait ConcreteArgs<A: Routine<RuntimeArgs>>:
    MatmulArgs<
        Input<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>: ConcreteInputsFactory<A>,
        Output<Vector<Acc, AccSize>>: ConcreteOutputFactory<A>,
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
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_n = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.n = shape_n;
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
        let channel_align = match blueprint.swizzle_modes.rhs {
            SwizzleMode::None => blueprint.tiling_scheme.tile_size.n() as usize,
            _ => blueprint.tiling_scheme.elements_per_stage_along_n() as usize,
        };
        let padded_channels = problem.channels.next_multiple_of(channel_align);
        let shape_n = problem.kernel_size.iter().product::<u32>() as usize * padded_channels;

        problem.n = shape_n;
        problem.padded_channels = padded_channels;

        problem
    }
}

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory<A: Routine<RuntimeArgs>>: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<R: Runtime>(
        input: MatmulInputBinding<R>,
        out_grad: MatmulInputBinding<R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<R>, RuntimeArgsLaunch<R>);
}

/// Create the output runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory<A: Routine<RuntimeArgs>>: LaunchArg {
    fn create<R: Runtime>(
        out: TensorBinding<R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
    ) -> Self::RuntimeArg<R>;
}

impl<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive, A: Routine<RuntimeArgs>>
    ConcreteInputsFactory<A> for TensorInputs<Lhs, Rhs, EO>
{
    fn create<R: Runtime>(
        input: MatmulInputBinding<R>,
        out_grad: MatmulInputBinding<R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
        _dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<R>, RuntimeArgsLaunch<R>) {
        type LhsLayout = Chain<NhwcLayout, Transpose<OutLayout>>;
        type RhsLayout = Chain<NhwcLayout, Im2colLayout>;

        let padded_channels = problem.padded_channels as u32;
        let params = ConvolutionParams::from_problem(problem);

        let layout_lhs = OutLayoutLaunch::from_args(problem, blueprint.lhs_global_layout_config());
        let layout_rhs =
            Im2colLayoutLaunch::from_args(problem, params, blueprint.rhs_global_layout_config());

        let layout_lhs = {
            let global = NhwcLayoutLaunch::unchecked();
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
            let global = NhwcLayoutLaunch::checked(checks);
            ChainLaunch::new(global, layout_rhs)
        };

        let inputs = TensorInputsLaunch::new(
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new_tensor::<LhsLayout>(out_grad.into_data().into_tensor_arg(), layout_lhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new_tensor::<RhsLayout>(input.into_data().into_tensor_arg(), layout_rhs),
            ComptimeOptionArgs::None,
            ComptimeOptionArgs::None,
        );

        let runtime_args = RuntimeArgsLaunch::new(
            problem.k as u32,
            problem.channels as u32,
            padded_channels,
            problem.operation,
        );

        (inputs, runtime_args)
    }
}

impl<EG: CubePrimitive, A: Routine<RuntimeArgs>> ConcreteOutputFactory<A> for TensorOutput<EG> {
    fn create<R: Runtime>(
        out: TensorBinding<R>,
        blueprint: &A::Blueprint,
        problem: &ConvolutionProblem,
    ) -> Self::RuntimeArg<R> {
        // Weight layout assumes col-major so it's technically "transposed" when it's row-major.
        // Should look into maybe inverting this and using `Transpose` for forward instead.
        type Layout = Chain<NhwcLayout, Transpose<WeightLayout>>;

        let mut checks = EnumSet::empty();
        if problem.should_check_channel() {
            checks.insert(NhwcCheck::Channel);
        }
        let global = NhwcLayoutLaunch::checked(checks);
        let layout = WeightLayoutLaunch::from_args(problem, blueprint.out_global_layout_config());
        let layout = ChainLaunch::new(global, TransposeLaunch::new(layout));
        let view = ViewArg::new_tensor::<Layout>(out.into_tensor_arg(), layout);
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
        input: MatmulInputBinding<R>,
        out_grad: MatmulInputBinding<R>,
        blueprint: &TilingBlueprint,
        problem: &ConvolutionProblem,
        dtypes: &MatmulElems,
    ) -> (Self::RuntimeArg<R>, RuntimeArgsLaunch<R>) {
        type LhsLayout = Transpose<TmaOutGradLayout>;
        type RhsLayout = TmaIm2colLayout;

        let tiling_scheme = blueprint.tiling_scheme;
        let stage_m = tiling_scheme.elements_per_stage_along_m();
        let stage_n = tiling_scheme.elements_per_stage_along_n();
        let stage_k = tiling_scheme.elements_per_stage_along_k();
        let tile_size_m = tiling_scheme.tile_size.m;
        let tile_size_n = tiling_scheme.tile_size.n;

        let dim_c = out_grad.shape().len() - 1;
        let stage_size_lhs = match blueprint.swizzle_modes.lhs {
            SwizzleMode::None => shape![stage_k as usize, tile_size_m as usize],
            _ => shape![stage_k as usize, stage_m as usize],
        };

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if dtypes.lhs_stage == f32::as_type_native_unchecked().storage_type() {
            tf32::as_type_native_unchecked().storage_type()
        } else {
            dtypes.lhs_stage
        };
        let rhs_elem = if dtypes.rhs_stage == f32::as_type_native_unchecked().storage_type() {
            tf32::as_type_native_unchecked().storage_type()
        } else {
            dtypes.rhs_stage
        };

        let mut elem_stride = strides![1; 2 + problem.stride.len()];

        for (i, stride) in problem.stride.iter().enumerate() {
            elem_stride[i + 1] = *stride as usize;
        }

        let lhs_shape = shape![problem.k, problem.m];
        let lhs_strides = strides![
            out_grad.data().strides[dim_c - 1],
            out_grad.data().strides[dim_c],
        ];

        let lhs_meta = TensorMapMeta {
            format: TensorMapFormat::Tiled(TiledArgs {
                tile_size: stage_size_lhs,
            }),
            metadata: Metadata::new(lhs_shape, lhs_strides),
            elem_stride: strides![1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: blueprint.swizzle_modes.lhs.into(),
            prefetch: TensorMapPrefetch::None,
            oob_fill: OobFill::Zero,
            storage_ty: lhs_elem,
        };

        let lhs = TensorMapArg {
            tensor: out_grad.clone().into_data().into_tensor_arg(),
            metadata: lhs_meta,
            _kind: core::marker::PhantomData,
        };

        let channels_per_pixel = match blueprint.swizzle_modes.rhs {
            SwizzleMode::None => tile_size_n,
            _ => stage_n,
        };

        let rhs = TensorMapArg::new(
            Im2colArgs {
                pixel_box_lower_corner: calculate_lower_corner(&problem.padding),
                pixel_box_upper_corner: calculate_upper_corner(
                    &problem.padding,
                    &problem.kernel_size,
                    &problem.dilation,
                ),
                channels_per_pixel,
                pixels_per_column: stage_k,
            },
            input.into_data().into_tensor_arg(),
            rhs_elem,
        )
        .with_elem_stride(elem_stride)
        .with_swizzle(blueprint.swizzle_modes.rhs.into());

        let padded_channels = problem.padded_channels as u32;
        let shape_k = problem.k as u32;
        let shape_n = problem.n as u32;

        // Im2col needs extra checking because if `n` is OOB it wraps around the kernel and can load
        // in-bounds but not in-kernel elements. Other TMA layouts are always outside the shape if
        // any matrix dim is out of bounds.
        let stages_rhs = A::num_stages().rhs;
        let stages_size_n = blueprint.tiling_scheme.elements_per_stage_along_n() * stages_rhs;

        let lhs_layout = TmaOutGradLayoutLaunch::from_problem(problem);
        let lhs_layout = TransposeLaunch::new(lhs_layout);

        let check_kernel = !shape_n.is_multiple_of(stages_size_n);
        let rhs_layout = TmaIm2colLayoutLaunch::from_args(problem, check_kernel);

        let inputs = TensorMapInputsLaunch::new(
            ViewArg::new_tensor_map_tiled::<LhsLayout>(lhs, lhs_layout),
            ViewArg::new_tensor_map_im2col::<RhsLayout, _, _>(rhs, rhs_layout),
            ComptimeOptionArgs::None,
            ComptimeOptionArgs::None,
        );

        let runtime_args = RuntimeArgsLaunch::new(
            shape_k,
            problem.channels as u32,
            padded_channels,
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
