use std::marker::PhantomData;

use cubecl::std::tensor::{
    View,
    launch::ViewArg,
    layout::{Coords1d, VirtualLayout, VirtualLayoutLaunch},
};
use cubecl::{
    prelude::*,
    zspace::{metadata::Metadata, shape, strides},
};
use cubecl::{server::TensorMapMeta, unexpanded};
use cubek_std::{MatrixLayout, stage::SwizzleMode};

use crate::components::global::memory::{
    BatchLayout, BatchLayoutLaunch, GlobalLayout, GlobalLayoutConfig, GlobalLayoutLaunch,
    GlobalScaleLayout, NoopLayout, NoopLayoutLaunch, SimpleTmaGlobalLayout,
    SimpleTmaGlobalLayoutLaunch,
};
use crate::definition::{
    Blueprint as _, MatmulElems, MatmulLineSizes, MatmulProblem, TilingBlueprint,
};
use crate::launch::handle::MatmulInputBinding;
use crate::routines::Routine;

/// Input argument
pub type InputArg<MA> =
    <MA as MatmulArgs>::Input<NumericExpand<0>, NumericExpand<1>, NumericExpand<2>>;

/// Output argument
pub type OutputArg<MA> = <MA as MatmulArgs>::Output<NumericExpand<2>>;

/// Config argument
pub type ConfigArg<MA> = <MA as MatmulArgs>::Config;

/// Input runtime argument
pub type InputRuntimeArg<'a, MA, R> = <InputArg<MA> as LaunchArg>::RuntimeArg<'a, R>;

/// Config runtime argument
pub type ConfigRuntimeArg<'a, MA, R> = <ConfigArg<MA> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MA, R> = <OutputArg<MA> as LaunchArg>::RuntimeArg<'a, R>;

pub type BatchedCoords = (usize, u32, u32);

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory<A: Routine<()>>: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<'a, R: Runtime>(
        client: &'a ComputeClient<R>,
        lhs: MatmulInputBinding<R>,
        rhs: MatmulInputBinding<R>,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime argument for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory<A: Routine<()>>: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: TensorBinding<R>,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R>;
}

pub trait RuntimeConfig: LaunchArg + CubeType + Clone + Send + Sync {}
impl<T: LaunchArg + CubeType + Clone + Send + Sync> RuntimeConfig for T {}

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric>: LaunchArg + CubeType;

    /// Type used for the output.
    type Output<EO: Numeric>: LaunchArg + CubeType;

    /// Type used for runtime configuration.
    type Config: RuntimeConfig;

    /// Inner state that is used to create tensor inputs and
    /// tensor outputs.
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric>: CubeType;

    /// Init the state.
    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] lhs_layout_config: GlobalLayoutConfig,
        #[comptime] rhs_layout_config: GlobalLayoutConfig,
        #[comptime] out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO>;

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, BatchedCoords> {
        unexpanded!()
    }
    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }
    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, BatchedCoords> {
        unexpanded!()
    }
    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }
    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<Line<EO>, BatchedCoords>> {
        unexpanded!()
    }
    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }
    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, BatchedCoords, ReadWrite> {
        unexpanded!()
    }
    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }

    fn runtime_config<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        unexpanded!()
    }
}

#[derive(Clone, Copy)]
/// Identification of the tensor input.
pub enum TensorInputIdent {
    Lhs,
    Rhs,
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorArgs<Config: RuntimeConfig = ()> {
    _config: PhantomData<Config>,
}

#[derive(CubeLaunch, CubeType, Clone, Copy)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<Lhs: Numeric, Rhs: Numeric, Acc: Numeric> {
    /// The lhs tensor.
    lhs_batch: VirtualLayout<Coords1d, Coords1d>,
    lhs: View<Line<Lhs>, BatchedCoords>,
    /// The rhs tensor.
    rhs_batch: VirtualLayout<Coords1d, Coords1d>,
    rhs: View<Line<Rhs>, BatchedCoords>,
    /// The tensor for loading the accumulator, if present
    acc_batch: ComptimeOption<VirtualLayout<Coords1d, Coords1d>>,
    acc: ComptimeOption<View<Line<Acc>, BatchedCoords>>,
}

impl<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, A: Routine<()>> ConcreteInputsFactory<A>
    for TensorInputs<Lhs, Rhs, Acc>
{
    fn create<'a, R: Runtime>(
        client: &'a ComputeClient<R>,
        lhs: MatmulInputBinding<R>,
        rhs: MatmulInputBinding<R>,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        let view =
            |handle: MatmulInputBinding<R>, config: GlobalLayoutConfig, line_size| match handle {
                MatmulInputBinding::Normal(handle, _dtype) => {
                    let layout = GlobalLayoutLaunch::from_handle(&handle, line_size, config);
                    ViewArg::new::<GlobalLayout>(
                        handle.into_tensor_arg(line_size).into_array_arg(),
                        layout,
                    )
                }
                MatmulInputBinding::Quantized {
                    data,
                    scale,
                    shape,
                    scheme,
                    ..
                } => {
                    let (data_layout, scales_layout) = GlobalLayoutLaunch::from_quantized_handle(
                        client, &data, &scale, &shape, problem, scheme, line_size, config,
                    );
                    let data_view = ViewArg::new::<GlobalLayout>(
                        data.into_tensor_arg(line_size).into_array_arg(),
                        data_layout,
                    );
                    let scales_view =
                        ViewArg::new::<GlobalScaleLayout>(scale.into_array_arg(1), scales_layout);
                    ViewArg::new_quantized(data_view, scales_view, scheme)
                }
            };
        let batch_layout = |handle: &MatmulInputBinding<R>| match handle {
            MatmulInputBinding::Normal(handle, _dtype) => {
                let layout = BatchLayoutLaunch::from_handle(client, handle, problem);
                VirtualLayoutLaunch::new::<BatchLayout>(layout)
            }
            MatmulInputBinding::Quantized { .. } => {
                VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new())
            }
        };

        TensorInputsLaunch::new(
            batch_layout(&lhs),
            view(lhs, blueprint.lhs_global_layout_config(), line_sizes.lhs),
            batch_layout(&rhs),
            view(rhs, blueprint.rhs_global_layout_config(), line_sizes.rhs),
            ComptimeOptionArgs::None,
            ComptimeOptionArgs::None,
        )
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct TensorOutput<EG: Numeric> {
    view: View<Line<EG>, BatchedCoords, ReadWrite>,
    batch: VirtualLayout<Coords1d, Coords1d>,
}

impl<EG: Numeric, A: Routine<()>> ConcreteOutputFactory<A> for TensorOutput<EG> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: TensorBinding<R>,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        let layout = GlobalLayoutLaunch::from_handle(
            &out,
            line_sizes.out,
            blueprint.out_global_layout_config(),
        );
        let batch = BatchLayoutLaunch::from_handle(client, &out, problem);
        let view = ViewArg::new::<GlobalLayout>(out.into_array_arg(line_sizes.out), layout);
        TensorOutputLaunch::new(view, VirtualLayoutLaunch::new::<BatchLayout>(batch))
    }
}

#[cube]
impl<Config: RuntimeConfig> MatmulArgs for TensorArgs<Config> {
    type Output<EO: Numeric> = TensorOutput<EO>;
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorInputs<Lhs, Rhs, EO>;
    type Config = Config;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> =
        (TensorInputs<Lhs, Rhs, EO>, TensorOutput<EO>, Config);

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] _lhs_layout_config: GlobalLayoutConfig,
        #[comptime] _rhs_layout_config: GlobalLayoutConfig,
        #[comptime] _out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO> {
        (*input, *output, config)
    }

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, BatchedCoords> {
        state.0.lhs
    }

    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.0.lhs_batch.to_source_pos(batch)
    }

    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, BatchedCoords> {
        state.0.rhs
    }

    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.0.rhs_batch.to_source_pos(batch)
    }

    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<Line<EO>, BatchedCoords>> {
        state.0.acc
    }

    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        #[comptime]
        match state.0.acc_batch {
            ComptimeOption::Some(layout) => layout.to_source_pos(batch),
            ComptimeOption::None => batch,
        }
    }

    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, BatchedCoords, ReadWrite> {
        state.1.view
    }

    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.1.batch.to_source_pos(batch)
    }

    fn runtime_config<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        state.2.clone()
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensor maps.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorMapArgs<Config: RuntimeConfig = ()> {
    _config: PhantomData<Config>,
}

#[derive(CubeLaunch, CubeType, Clone, Copy)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorMapInputs<Lhs: Numeric, Rhs: Numeric, EO: Numeric> {
    /// The lhs tensor.
    pub lhs: View<Line<Lhs>, BatchedCoords>,
    /// The rhs tensor.
    pub rhs: View<Line<Rhs>, BatchedCoords>,
    /// The accumulator
    pub acc: ComptimeOption<View<Line<EO>, BatchedCoords>>,
    /// The accumulator batch layout
    pub acc_batch: ComptimeOption<VirtualLayout<Coords1d, Coords1d>>,
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, A: Routine<(), Blueprint = TilingBlueprint>>
    ConcreteInputsFactory<A> for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        _client: &'a ComputeClient<R>,
        lhs_handle: MatmulInputBinding<R>,
        rhs_handle: MatmulInputBinding<R>,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        let lhs = lhs_handle.into_data();
        let rhs = rhs_handle.into_data();

        let tiling_scheme = blueprint.tiling_scheme;
        let stage_m = tiling_scheme.elements_per_stage_along_m();
        let stage_n = tiling_scheme.elements_per_stage_along_n();
        let stage_k = tiling_scheme.elements_per_stage_along_k();

        // Loaders use dynamic layout based on swizzle setting. For no swizzle, contiguous tiles are
        // loaded and TMA loads single tile wide columns.
        // For swizzled, bank conflicts aren't an issue so the tile size is the full stage.
        let stage_size_lhs = match blueprint.swizzle_modes.lhs {
            SwizzleMode::None => match problem.lhs_layout {
                MatrixLayout::RowMajor => {
                    shape![1, stage_m as usize, tiling_scheme.tile_size.k as usize]
                }
                MatrixLayout::ColMajor => {
                    shape![1, stage_k as usize, tiling_scheme.tile_size.m as usize]
                }
            },
            _ => match problem.lhs_layout {
                MatrixLayout::RowMajor => {
                    shape![1, stage_m as usize, stage_k as usize]
                }
                MatrixLayout::ColMajor => {
                    shape![1, stage_k as usize, stage_m as usize]
                }
            },
        };
        let stage_size_rhs = match blueprint.swizzle_modes.rhs {
            SwizzleMode::None => match problem.rhs_layout {
                MatrixLayout::RowMajor => {
                    shape![1, stage_k as usize, tiling_scheme.tile_size.n as usize]
                }
                MatrixLayout::ColMajor => {
                    shape![1, stage_n as usize, tiling_scheme.tile_size.k as usize]
                }
            },
            _ => match problem.rhs_layout {
                MatrixLayout::RowMajor => {
                    shape![1, stage_k as usize, stage_n as usize]
                }
                MatrixLayout::ColMajor => {
                    shape![1, stage_n as usize, stage_k as usize]
                }
            },
        };

        let lhs_rank = lhs.shape.len();
        let mut lhs_shape = shape![
            problem.lhs_batches.iter().product(),
            lhs.shape[lhs_rank - 2],
            lhs.shape[lhs_rank - 1],
        ];
        let mut lhs_strides = if lhs_rank > 2 {
            lhs.strides[lhs_rank - 3..].into()
        } else {
            strides![lhs.strides[0], lhs.strides[1]]
        };

        let rhs_rank = rhs.shape.len();
        let mut rhs_shape = shape![
            problem.rhs_batches.iter().product(),
            rhs.shape[rhs_rank - 2],
            rhs.shape[rhs_rank - 1],
        ];
        let mut rhs_strides = if rhs_rank > 2 {
            rhs.strides[rhs_rank - 3..].into()
        } else {
            strides![rhs.strides[0], rhs.strides[1]]
        };

        let mut lhs_transposed = false;
        let mut rhs_transposed = false;

        let lhs_rank = lhs_strides.len();
        let rhs_rank = rhs_strides.len();

        // TMA assumes the last stride is contiguous and won't even take it, so we need to map it
        // with transposed shape and stride. Tensor metadata still has the normal layout.
        if matches!(problem.lhs_layout, MatrixLayout::ColMajor) {
            lhs_shape.swap(2, 1);
            lhs_strides.swap(lhs_rank - 1, lhs_rank - 2);
            lhs_transposed = true;
        }
        if matches!(problem.rhs_layout, MatrixLayout::ColMajor) {
            rhs_shape.swap(2, 1);
            rhs_strides.swap(rhs_rank - 1, rhs_rank - 2);
            rhs_transposed = true;
        }

        // Insert batch stride after swap so we can easily get the non-contiguous stride
        if lhs_rank == 2 {
            let stride = lhs_strides[0];
            lhs_strides.insert(0, stride);
        }
        if rhs_rank == 2 {
            let stride = rhs_strides[0];
            rhs_strides.insert(0, stride);
        }

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if dtypes.lhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            dtypes.lhs_stage
        };
        let rhs_elem = if dtypes.rhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            dtypes.rhs_stage
        };

        let meta_lhs = TensorMapMeta {
            format: TensorMapFormat::Tiled(TiledArgs {
                tile_size: stage_size_lhs,
            }),
            metadata: Metadata::new(lhs_shape.clone(), lhs_strides),
            elem_stride: strides![1, 1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: blueprint.swizzle_modes.lhs.into(),
            prefetch: TensorMapPrefetch::None,
            oob_fill: OobFill::Zero,
            storage_ty: lhs_elem,
        };

        let meta_rhs = TensorMapMeta {
            format: TensorMapFormat::Tiled(TiledArgs {
                tile_size: stage_size_rhs,
            }),
            metadata: Metadata::new(rhs_shape.clone(), rhs_strides),
            elem_stride: strides![1, 1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: blueprint.swizzle_modes.rhs.into(),
            prefetch: TensorMapPrefetch::None,
            oob_fill: OobFill::Zero,
            storage_ty: rhs_elem,
        };

        let lhs = TensorMapArg {
            tensor: lhs.into_tensor_arg(line_sizes.lhs),
            metadata: meta_lhs,
            _kind: PhantomData,
        };
        let rhs = TensorMapArg {
            tensor: rhs.into_tensor_arg(line_sizes.rhs),
            metadata: meta_rhs,
            _kind: PhantomData,
        };

        let view = |buffer, shape: &[usize], transposed| {
            let batches = ScalarArg::new(shape[0]);
            let (rows, cols) = match transposed {
                true => (
                    ScalarArg::new(shape[2] as u32),
                    ScalarArg::new(shape[1] as u32),
                ),
                false => (
                    ScalarArg::new(shape[1] as u32),
                    ScalarArg::new(shape[2] as u32),
                ),
            };
            let shape = (batches, rows, cols);
            let layout = SimpleTmaGlobalLayoutLaunch::new(transposed, shape);
            ViewArg::new_tensor_map_tiled::<SimpleTmaGlobalLayout>(buffer, layout)
        };

        TensorMapInputsLaunch::new(
            view(lhs, &lhs_shape, lhs_transposed),
            view(rhs, &rhs_shape, rhs_transposed),
            ComptimeOptionArgs::None,
            ComptimeOptionArgs::None,
        )
    }
}

#[cube]
impl<Config: RuntimeConfig> MatmulArgs for TensorMapArgs<Config> {
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorMapInputs<Lhs, Rhs, EO>;
    type Output<EO: Numeric> = TensorOutput<EO>;
    type Config = Config;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> =
        (TensorMapInputs<Lhs, Rhs, EO>, TensorOutput<EO>, Config);

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] _lhs_layout_config: GlobalLayoutConfig,
        #[comptime] _rhs_layout_config: GlobalLayoutConfig,
        #[comptime] _out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO> {
        (*input, *output, config)
    }

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, BatchedCoords> {
        state.0.lhs
    }

    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        batch
    }

    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, BatchedCoords> {
        state.0.rhs
    }

    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        batch
    }

    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<Line<EO>, BatchedCoords>> {
        state.0.acc
    }

    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        #[comptime]
        match state.0.acc_batch {
            ComptimeOption::Some(layout) => layout.to_source_pos(batch),
            ComptimeOption::None => batch,
        }
    }

    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, BatchedCoords, ReadWrite> {
        state.1.view
    }

    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.1.batch.to_source_pos(batch)
    }

    fn runtime_config<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        state.2.clone()
    }
}
