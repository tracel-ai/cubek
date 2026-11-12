//! Split the implicit weight-gradient contraction without materializing windows.

use cubecl::{
    calculate_cube_count_elemwise,
    ir::EnumSet,
    prelude::*,
    std::{
        FastDivmod,
        tensor::{
            TensorHandle,
            launch::ViewArg,
            layout::{
                Layout, LayoutExpand, VirtualLayoutLaunch,
                chain::{Chain, ChainLaunch},
            },
        },
    },
};
use cubek_matmul::{
    definition::{AccumulatorOperand, AvailableVectorSizes, MatmulElems, MatmulProblem},
    multi_level::{
        BatchMatmulRoutine,
        args::{BatchedCoords, TensorArgs, TensorInputsLaunch, TensorOutputLaunch},
        components::global::memory::{
            BatchLayout, BatchLayoutLaunch, GlobalLayout, GlobalLayoutLaunch, NoopLayout,
            NoopLayoutLaunch, Transpose, TransposeLaunch,
        },
        components::tile::TileMatmulKind,
        definition::Blueprint,
        routines::batch::simple::SimpleArgs,
    },
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    ConvolutionArgs, UnitStrategy,
    components::{
        ConvSetupError, ConvolutionOperation, ConvolutionParams, ConvolutionProblem,
        Dimensionality,
        global::{
            args::{RuntimeArgs, RuntimeArgsLaunch},
            layout::{
                Im2colLayout, Im2colLayoutLaunch, NhwcCheck, NhwcLayout, NhwcLayoutLaunch,
                OutLayout, OutLayoutLaunch,
            },
        },
    },
    routines::{Routine, simple::SimpleSyncStridedConv, unit::UnitConv},
};

/// Reduction rows one split should cover at least; fewer splits leave the
/// device underused on large feature maps, more pay for the extra partials.
const ROWS_PER_SPLIT: usize = 2048;
const MAX_SPLITS: usize = 64;
/// Bounds the partial matrices independently of the input's spatial extent.
const MAX_PARTIAL_BYTES: usize = 128 * 1024 * 1024;

/// How many disjoint reduction intervals a weight gradient over `rows`
/// output positions is cut into: a power of two, one when splitting cannot pay.
pub(crate) fn split_count(rows: usize) -> usize {
    let by_rows = (rows / ROWS_PER_SPLIT).clamp(1, MAX_SPLITS);
    1 << by_rows.ilog2()
}

struct SplitOptions<Build> {
    blueprint: Build,
    rows_per_split: usize,
}

/// Ordering of spatial taps and input channels within a partial matrix column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PartialColumnOrder {
    /// Input channels vary fastest, inside the spatial axes.
    ChannelsLast,
    /// Spatial axes vary fastest, inside the input channel axis.
    SpatialLast,
}

/// Tile implementation and partial column ordering for accelerated split weight gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcceleratedSplitStrategy {
    /// Accelerated tile implementation used for each partial matrix.
    pub tile_kind: crate::AcceleratedTileKind,
    /// Ordering of spatial taps and input channels in partial matrices.
    pub column_order: PartialColumnOrder,
}

/// Launches register-tiled split weight gradients for NHWC tensors of rank `N + 2`.
pub(crate) fn launch_unit<const N: usize>(
    client: &Client,
    input: InputBinding,
    out_grad: InputBinding,
    weight_grad: TensorBinding,
    args: ConvolutionArgs<N>,
    strategy: UnitStrategy,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let dtype = f32::elem_type_native();
    if dtypes.lhs_global != dtype || dtypes.rhs_global != dtype || dtypes.acc_global != dtype {
        return Err(ConvSetupError::Unknown);
    }
    launch_split::<N, UnitConv, _>(
        client,
        input,
        out_grad,
        weight_grad,
        args,
        PartialColumnOrder::ChannelsLast,
        SplitOptions {
            blueprint: |problem: &MatmulProblem, dtypes: &MatmulElems| {
                strategy.blueprint(client, problem, dtypes)
            },
            rows_per_split: ROWS_PER_SPLIT,
        },
        dtypes,
    )
}

/// Launches accelerated split weight gradients for NHWC input and output gradients.
/// The output has shape `[out_channels, ...kernel, in_channels]`, with rank `N + 2`.
/// F16 and BF16 partial sums stay F32 until the final output write.
pub fn launch_accelerated<const N: usize>(
    client: &Client,
    input: InputBinding,
    out_grad: InputBinding,
    weight_grad: TensorBinding,
    args: ConvolutionArgs<N>,
    strategy: AcceleratedSplitStrategy,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_split::<N, SimpleSyncStridedConv, _>(
        client,
        input,
        out_grad,
        weight_grad,
        args,
        strategy.column_order,
        SplitOptions {
            blueprint: |_: &MatmulProblem, _: &MatmulElems| {
                cubek_matmul::routine::BlueprintStrategy::Inferred(SimpleArgs {
                    tile_matmul: match strategy.tile_kind {
                        crate::AcceleratedTileKind::Cmma => TileMatmulKind::Cmma,
                        crate::AcceleratedTileKind::Mma => TileMatmulKind::Mma,
                    },
                    multi_rows: false,
                    plane_tiling: None,
                })
            },
            rows_per_split: 512,
        },
        dtypes,
    )
}

/// Checks logical NHWC shapes and the signed coordinates used by convolution layouts.
fn coordinates_fit<const N: usize>(
    input: &InputBinding,
    out_grad: &InputBinding,
    weight_grad: &TensorBinding,
    args: &ConvolutionArgs<N>,
) -> bool {
    if !(1..=3).contains(&N) {
        return false;
    }
    let tensors = [input.data(), out_grad.data(), weight_grad];
    if tensors.iter().any(|tensor| {
        tensor.shape.len() != N + 2
            || tensor.strides.len() != N + 2
            || tensor.tiling.is_tiled()
            || tensor.shape.contains(&0)
            || tensor
                .shape
                .iter()
                .try_fold(1usize, |count, size| count.checked_mul(*size))
                .is_none_or(|count| count > u32::MAX as usize)
    }) {
        return false;
    }
    if input.shape()[0] != out_grad.shape()[0]
        || input.shape()[N + 1] != weight_grad.shape[N + 1]
        || out_grad.shape()[N + 1] != weight_grad.shape[0]
    {
        return false;
    }
    let mut rows = input.shape()[0] as u128;
    (0..N).all(|axis| {
        let extent = cubecl::tune::anchor(input.shape()[axis + 1], None, None, None) as u128;
        let output = cubecl::tune::anchor(out_grad.shape()[axis + 1], None, None, None) as u128;
        let kernel = weight_grad.shape[axis + 1] as u128;
        let stride = args.stride[axis] as u128;
        let dilation = args.dilation[axis] as u128;
        let padding = args.padding[axis] as u128;
        let limit = i32::MAX as u128;
        rows *= output;
        stride > 0
            && dilation > 0
            && padding <= limit
            && extent <= limit + 1
            && (output - 1) * stride + (kernel - 1) * dilation <= limit
            && rows <= u32::MAX as u128
    })
}

/// Each split writes its own small matrix, followed by a deterministic sum.
/// The final split may be short; its missing rows are masked in `SplitKLayout`.
#[allow(clippy::too_many_arguments, clippy::result_large_err)]
fn launch_split<const N: usize, Rt, Build>(
    client: &Client,
    input: InputBinding,
    out_grad: InputBinding,
    weight_grad: TensorBinding,
    args: ConvolutionArgs<N>,
    column_order: PartialColumnOrder,
    options: SplitOptions<Build>,
    mut dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Rt: Routine<Args = TensorArgs<RuntimeArgs>>,
    Build: FnOnce(
        &MatmulProblem,
        &MatmulElems,
    ) -> cubek_matmul::routine::BlueprintStrategy<Rt::MatmulRoutine>,
{
    let dtype = f32::elem_type_native();
    let output_dtype = dtypes.acc_global;
    if input.scale().is_some() || out_grad.scale().is_some() {
        return Err(ConvSetupError::Unknown);
    }
    if !coordinates_fit(&input, &out_grad, &weight_grad, &args) {
        return Err(ConvSetupError::InvalidConfig(Box::new(
            "split weight-gradient shapes or coordinates exceed the layout index range",
        )));
    }
    let supported = [
        f32::elem_type_native(),
        half::f16::elem_type_native(),
        half::bf16::elem_type_native(),
    ];
    if !supported.contains(&dtypes.lhs_global)
        || !supported.contains(&dtypes.rhs_global)
        || !supported.contains(&output_dtype)
    {
        return Err(ConvSetupError::Unknown);
    }
    dtypes.acc_global = dtype;
    dtypes.acc_stage = dtype;
    dtypes.acc_register = dtype;
    let op = ConvolutionOperation::BackwardWeight;
    let spatial_last = column_order == PartialColumnOrder::SpatialLast;
    let input = if spatial_last {
        input.into_data()
    } else {
        Rt::correct_layout(client, input.into_data(), dtypes.rhs_global, op)?
    };
    let out_grad = Rt::correct_layout(client, out_grad.into_data(), dtypes.lhs_global, op)?;
    let channels = input.shape[N + 1];
    let out_channels = out_grad.shape[N + 1];
    let rows = out_grad.shape[..N + 1].iter().product::<usize>();
    let taps = weight_grad.shape[1..N + 1].iter().product::<usize>();
    let padded_channels = channels
        .checked_next_multiple_of(
            (client.properties().io_width() as usize / dtypes.rhs_global.size_bits()).max(1),
        )
        .ok_or(ConvSetupError::Unknown)?;
    let width = taps
        .checked_mul(padded_channels)
        .ok_or(ConvSetupError::Unknown)?;
    let by_rows = (rows / options.rows_per_split).clamp(1, MAX_SPLITS);
    let mut split = 1usize << by_rows.ilog2();
    let Some(partial_bytes) = out_channels
        .checked_mul(width)
        .and_then(|n| n.checked_mul(4))
        .filter(|bytes| *bytes <= MAX_PARTIAL_BYTES)
    else {
        return Err(ConvSetupError::Unknown);
    };
    while split > 1 && split > MAX_PARTIAL_BYTES / partial_bytes {
        split /= 2;
    }
    let per = rows
        .div_ceil(split)
        .checked_next_multiple_of(32)
        .ok_or(ConvSetupError::Unknown)?;
    if rows > u32::MAX as usize || width > u32::MAX as usize || per > u32::MAX as usize {
        return Err(ConvSetupError::Unknown);
    }
    let partials = TensorHandle::empty(client, [split, out_channels, width], dtype);
    let problem = ConvolutionProblem {
        m: out_channels,
        n: width,
        k: rows,
        lhs_strides: out_grad.strides.clone(),
        rhs_strides: input.strides.clone(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: weight_grad.shape[1..N + 1]
            .iter()
            .map(|n| *n as u32)
            .collect(),
        stride: args.stride.map(|n| n as u32).to_vec(),
        padding: args.padding.map(|n| n as i32).to_vec(),
        dilation: args.dilation.map(|n| n as u32).to_vec(),
        batches: input.shape[0],
        channels,
        out_channels,
        in_shape: input.shape[1..N + 1].into(),
        out_shape: out_grad.shape[1..N + 1].into(),
        padded_channels,
        operation: op,
        dimensionality: match N {
            1 => Dimensionality::Dim1,
            2 => Dimensionality::Dim2,
            _ => Dimensionality::Dim3,
        },
        global_dtypes: dtypes.as_global_elems(),
        address_type: input
            .required_address_type(dtypes.rhs_global.size())
            .max(out_grad.required_address_type(dtypes.lhs_global.size()))
            .max(partials.clone().binding().required_address_type(4)),
    };
    let vector_sizes = Rt::filter_vector_sizes(AvailableVectorSizes::from_type_sizes(
        client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        4,
    ))
    .filter_lhs_with_tensor(&out_grad.strides, &out_grad.shape, MatrixLayout::RowMajor);
    let vector_sizes = if spatial_last {
        vector_sizes.filter_rhs(|size| {
            *size == 1
                || (input.strides[N] == 1
                    && input.shape[N].is_multiple_of(*size)
                    && args.dilation[N - 1] == 1
                    && weight_grad.shape[N].is_multiple_of(*size)
                    && args.stride[N - 1].is_multiple_of(*size)
                    && args.padding[N - 1].is_multiple_of(*size)
                    && input
                        .strides
                        .iter()
                        .enumerate()
                        .all(|(axis, stride)| axis == N || stride.is_multiple_of(*size)))
        })
    } else {
        vector_sizes.filter_rhs_with_tensor(&input.strides, &input.shape, MatrixLayout::RowMajor)
    };
    let vector_sizes = vector_sizes
        .filter_out_with_tensor(partials.strides(), partials.shape())
        .pick_max()?;
    let mut matmul_problem = problem.as_matmul_problem(AccumulatorOperand::Absent);
    matmul_problem.k = per;
    matmul_problem.lhs_batches = vec![split].into();
    matmul_problem.rhs_batches = vec![split].into();
    matmul_problem.out_batches = vec![split].into();
    matmul_problem.lhs_shape = vec![split, out_channels, per].into();
    matmul_problem.rhs_shape = vec![split, per, width].into();
    matmul_problem.out_shape = vec![split, out_channels, width].into();
    matmul_problem.lhs_strides = vec![
        per.checked_mul(out_channels)
            .ok_or(ConvSetupError::Unknown)?,
        1,
        out_channels,
    ]
    .into();
    matmul_problem.rhs_strides = vec![
        per.checked_mul(width).ok_or(ConvSetupError::Unknown)?,
        width,
        1,
    ]
    .into();
    matmul_problem.out_strides = partials.strides().into();

    let settings = <Rt::MatmulRoutine as BatchMatmulRoutine<RuntimeArgs>>::device_settings(
        client,
        vector_sizes,
    );
    let strategy = (options.blueprint)(&matmul_problem, &dtypes);
    let expand = <Rt::MatmulRoutine as BatchMatmulRoutine<RuntimeArgs>>::expand_blueprint(
        &matmul_problem,
        &settings,
        &strategy,
    )?;
    let launch = <Rt::MatmulRoutine as BatchMatmulRoutine<RuntimeArgs>>::prepare(
        &matmul_problem,
        &settings,
        expand,
    )?;
    let blueprint = &launch.blueprint;

    type LhsLayout = Chain<Chain<NhwcLayout, Transpose<OutLayout>>, SplitKLayout>;
    type RhsLayout = Chain<Chain<NhwcLayout, Im2colLayout>, SplitKLayout>;
    let lhs = ChainLaunch::new(
        ChainLaunch::new(
            NhwcLayoutLaunch::unchecked(),
            TransposeLaunch::new(OutLayoutLaunch::from_args(
                &problem,
                blueprint.lhs_global_layout_config(),
            )),
        ),
        SplitKLayoutLaunch::new(
            split,
            out_channels as u32,
            per as u32,
            rows as u32,
            per as u32,
            true,
            false,
            1,
            1,
        ),
    );
    let mut checks = EnumSet::empty();
    if problem.should_check_spatial_bounds() {
        checks.insert(NhwcCheck::Spatial);
    }
    if problem.should_check_channel() {
        checks.insert(NhwcCheck::Channel);
    }
    let rhs = ChainLaunch::new(
        ChainLaunch::new(
            NhwcLayoutLaunch::checked(checks),
            Im2colLayoutLaunch::from_args(
                &problem,
                ConvolutionParams::from_problem(&problem),
                blueprint.rhs_global_layout_config(),
            ),
        ),
        SplitKLayoutLaunch::new(
            split,
            per as u32,
            width as u32,
            rows as u32,
            per as u32,
            false,
            spatial_last,
            taps as u32,
            padded_channels as u32,
        ),
    );
    let inputs = TensorInputsLaunch::new(
        VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
        ViewArg::new_tensor::<LhsLayout>(out_grad.into_tensor_arg(), lhs),
        VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
        ViewArg::new_tensor::<RhsLayout>(input.into_tensor_arg(), rhs),
        ComptimeOptionArgs::None,
        ComptimeOptionArgs::None,
    );
    let partial_binding = partials.clone().binding();
    let layout = GlobalLayoutLaunch::from_handle(
        &partial_binding,
        vector_sizes.out,
        blueprint.out_global_layout_config(),
    );
    let batch = BatchLayoutLaunch::from_handle(&partial_binding, &matmul_problem);
    let output = TensorOutputLaunch::new(
        ViewArg::new_tensor::<GlobalLayout>(partial_binding.into_tensor_arg(), layout),
        VirtualLayoutLaunch::new::<BatchLayout>(batch),
    );
    let runtime = RuntimeArgsLaunch::new(per as u32, channels as u32, padded_channels as u32, op);
    cubek_matmul::multi_level::launch_kernel::<TensorArgs<RuntimeArgs>, Rt::MatmulRoutine>(
        client, inputs, output, runtime, launch,
    )?;

    let elements = weight_grad.shape.iter().product::<usize>();
    let cube_dim = CubeDim::new(client, elements);
    let address_type = partials
        .clone()
        .binding()
        .required_address_type(4)
        .max(weight_grad.required_address_type(output_dtype.size()));
    finish::launch(
        client,
        calculate_cube_count_elemwise(client, elements, cube_dim),
        cube_dim,
        address_type,
        partials.binding().into_tensor_arg(),
        weight_grad.into_tensor_arg(),
        channels,
        taps,
        padded_channels,
        elements,
        N,
        spatial_last,
        output_dtype,
    );
    Ok(())
}

/// The matmul batch axis selects a disjoint interval of the reduction axis.
#[derive(CubeType, CubeLaunch, Clone)]
struct SplitKLayout {
    splits: usize,
    rows: u32,
    cols: u32,
    total: u32,
    per: u32,
    #[cube(comptime)]
    lhs: bool,
    #[cube(comptime)]
    spatial_last: bool,
    taps: u32,
    channels: u32,
}

#[cube]
impl Layout for SplitKLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = BatchedCoords;

    fn to_source_pos(&self, pos: BatchedCoords) -> BatchedCoords {
        let (split, row, col) = pos;
        let offset = split as u32 * self.per;
        if self.lhs {
            (0, row, col + offset)
        } else {
            let col = if self.spatial_last.comptime() {
                (col % self.taps) * self.channels + col / self.taps
            } else {
                col
            };
            (0, row + offset, col)
        }
    }

    fn shape(&self) -> BatchedCoords {
        (self.splits, self.rows, self.cols)
    }

    fn is_in_bounds(&self, pos: BatchedCoords) -> bool {
        let (split, row, col) = pos;
        let k = if self.lhs { col } else { row };
        split < self.splits
            && row < self.rows
            && col < self.cols
            && split as u32 * self.per + k < self.total
    }

    fn to_source_pos_checked(&self, pos: BatchedCoords) -> (BatchedCoords, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

#[cube(launch, address_type = "dynamic")]
fn finish<Out: Float>(
    partials: &Tensor<f32>,
    weight_grad: &mut Tensor<Out>,
    channels: FastDivmod<usize>,
    taps: FastDivmod<usize>,
    padded_channels: usize,
    elements: usize,
    #[comptime] dims: usize,
    #[comptime] spatial_last: bool,
    #[define(Out)] _output_dtype: ElemType,
) {
    let index = ABSOLUTE_POS;
    if index >= elements {
        terminate!();
    }
    let (row, channel) = channels.div_mod(index);
    let (out_channel, tap) = taps.div_mod(row);
    let column = if spatial_last {
        channel * (partials.shape(2) / padded_channels) + tap
    } else {
        tap * padded_channels + channel
    };
    let base = out_channel * partials.stride(1) + column * partials.stride(2);
    let mut value = 0.0f32;
    for split in 0..partials.shape(0) {
        value += partials[split * partials.stride(0) + base];
    }
    let mut offset = out_channel * weight_grad.stride(0) + channel * weight_grad.stride(dims + 1);
    let mut tap = tap;
    #[unroll]
    for rev in 0..dims {
        let axis = dims - rev;
        offset += (tap % weight_grad.shape(axis)) * weight_grad.stride(axis);
        tap /= weight_grad.shape(axis);
    }
    weight_grad[offset] = Out::cast_from(value);
}
