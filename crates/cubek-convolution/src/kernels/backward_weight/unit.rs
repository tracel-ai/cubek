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
    definition::{AccumulatorOperand, AvailableVectorSizes, MatmulElems},
    multi_level::{
        BatchMatmulRoutine,
        args::{BatchedCoords, TensorArgs, TensorInputsLaunch, TensorOutputLaunch},
        components::global::memory::{
            BatchLayout, BatchLayoutLaunch, GlobalLayout, GlobalLayoutLaunch, NoopLayout,
            NoopLayoutLaunch, Transpose, TransposeLaunch,
        },
        definition::Blueprint,
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
    routines::{
        Routine,
        unit::{UnitConv, UnitMatmul},
    },
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

/// Each split writes its own small matrix, followed by a deterministic sum.
/// The final split may be short; its missing rows are masked in `SplitKLayout`.
#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub(crate) fn launch_split<const N: usize>(
    client: &Client,
    input: InputBinding,
    out_grad: InputBinding,
    weight_grad: TensorBinding,
    args: ConvolutionArgs<N>,
    strategy: UnitStrategy,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let dtype = f32::elem_type_native();
    if dtypes.lhs_global != dtype
        || dtypes.rhs_global != dtype
        || dtypes.acc_global != dtype
        || input.scale().is_some()
        || out_grad.scale().is_some()
    {
        return Err(ConvSetupError::Unknown);
    }
    let op = ConvolutionOperation::BackwardWeight;
    let input = UnitConv::correct_layout(client, input.into_data(), dtype, op)?;
    let out_grad = UnitConv::correct_layout(client, out_grad.into_data(), dtype, op)?;
    let channels = input.shape[N + 1];
    let out_channels = out_grad.shape[N + 1];
    let rows = out_grad.shape[..N + 1].iter().product::<usize>();
    let taps = weight_grad.shape[1..N + 1].iter().product::<usize>();
    let padded_channels = channels.next_multiple_of(client.properties().io_width() as usize / 32);
    let width = taps * padded_channels;
    let mut split = split_count(rows);
    let Some(partial_bytes) = out_channels
        .checked_mul(width)
        .and_then(|n| n.checked_mul(4))
        .filter(|bytes| *bytes <= MAX_PARTIAL_BYTES)
    else {
        return Err(ConvSetupError::Unknown);
    };
    while split > 1 && partial_bytes * split > MAX_PARTIAL_BYTES {
        split /= 2;
    }
    let per = rows.div_ceil(split).next_multiple_of(32);
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
            .required_address_type(4)
            .max(out_grad.required_address_type(4))
            .max(partials.clone().binding().required_address_type(4)),
    };
    let vector_sizes = UnitConv::filter_vector_sizes(
        AvailableVectorSizes::from_type_sizes(client, 4, 4, 4)
            .filter_lhs_with_tensor(&out_grad.strides, &out_grad.shape, MatrixLayout::RowMajor)
            .filter_rhs_with_tensor(&input.strides, &input.shape, MatrixLayout::RowMajor)
            .filter_out_with_tensor(partials.strides(), partials.shape()),
    )
    .pick_max()?;
    let mut matmul_problem = problem.as_matmul_problem(AccumulatorOperand::Absent);
    matmul_problem.k = per;
    matmul_problem.lhs_batches = vec![split].into();
    matmul_problem.rhs_batches = vec![split].into();
    matmul_problem.out_batches = vec![split].into();
    matmul_problem.lhs_shape = vec![split, out_channels, per].into();
    matmul_problem.rhs_shape = vec![split, per, width].into();
    matmul_problem.out_shape = vec![split, out_channels, width].into();
    matmul_problem.lhs_strides = vec![per * out_channels, 1, out_channels].into();
    matmul_problem.rhs_strides = vec![per * width, width, 1].into();
    matmul_problem.out_strides = partials.strides().into();

    let settings =
        <UnitMatmul as BatchMatmulRoutine<RuntimeArgs>>::device_settings(client, vector_sizes);
    let strategy = strategy.blueprint(client, &matmul_problem, &dtypes);
    let expand = UnitMatmul::expand_blueprint(&matmul_problem, &settings, &strategy)?;
    let launch = <UnitMatmul as BatchMatmulRoutine<RuntimeArgs>>::prepare(
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
    cubek_matmul::multi_level::launch_kernel::<TensorArgs<RuntimeArgs>, UnitMatmul>(
        client, inputs, output, runtime, launch,
    )?;

    let elements = weight_grad.shape.iter().product::<usize>();
    let cube_dim = CubeDim::new(client, elements);
    let address_type = partials
        .clone()
        .binding()
        .required_address_type(4)
        .max(weight_grad.required_address_type(4));
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
fn finish(
    partials: &Tensor<f32>,
    weight_grad: &mut Tensor<f32>,
    channels: FastDivmod<usize>,
    taps: FastDivmod<usize>,
    padded_channels: usize,
    elements: usize,
    #[comptime] dims: usize,
) {
    let index = ABSOLUTE_POS;
    if index >= elements {
        terminate!();
    }
    let (row, channel) = channels.div_mod(index);
    let (out_channel, tap) = taps.div_mod(row);
    let base =
        out_channel * partials.stride(1) + (tap * padded_channels + channel) * partials.stride(2);
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
    weight_grad[offset] = value;
}
