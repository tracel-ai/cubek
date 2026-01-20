use crate::{AcceleratedTileKind, ReadingStrategy};
use crate::{
    ConvolutionArgs, Strategy,
    components::{ConvGemmConfig as _, ConvolutionOperation},
    forward::args::ConcreteArgs,
    kernels::forward::simple::*,
};
use crate::{components::ConvSetupError, kernels::forward::selector::launch_kernel_concrete};
use crate::{
    components::{ConvolutionProblem, Dimensionality},
    kernels::forward::algorithm::Algorithm,
};
use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::*,
    std::{CubeOption, tensor::TensorHandle},
};
use cubek_matmul::launch::MatmulInputHandle;
use cubek_matmul::{
    components::tile::{cmma::CmmaMatmul, io::Strided, mma::MmaMatmul},
    definition::{AvailableLineSizes, MatmulElems, MatrixLayout},
};
use cubek_matmul::{definition, launch::MatmulInputHandleRef};
use derive_new::new;

macro_rules! with_tile_kind {
    ($kind: expr, $T: ident, $launch: expr) => {
        match $kind {
            AcceleratedTileKind::Cmma => {
                type $T = CmmaMatmul<CubeOption<Strided>>;
                ($launch)()
            }
            AcceleratedTileKind::Mma => {
                type $T = MmaMatmul<Strided, Strided, CubeOption<Strided>>;
                ($launch)()
            }
        }
    };
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch<R: Runtime, const N_SPATIAL: usize>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    input: MatmulInputHandle<R>,
    weight: MatmulInputHandle<R>,
    bias: Option<MatmulInputHandle<R>>,
    out: TensorHandle<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_ref(
        strategy,
        client,
        &input.as_ref(),
        &weight.as_ref(),
        &bias.as_ref().map(|it| it.as_ref()),
        &out.as_ref(),
        args,
        dtypes,
    )
}

/// Perform an n-dimensional convolution using the implicit GEMM (im2col) algorithm, using cubecl
/// tiling matmul components, using the specified algorithm.
///
/// * `input` - The input feature map, layout should be [batches, depth, height, width, in_channels]
/// * `weight` - The weights (filter) applied to each kernel, layout should be [out_channels, kernel_d, kernel_h, kernel_w, in_channels]
/// * `out` - The output feature map, layout should be [batches, out_depth, out_height, out_width, out_channels]
/// * `bias` - The bias added to each out channel
/// * `options` - The options to use for the convolution
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref<R: Runtime, const N_SPATIAL: usize>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<MatmulInputHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let conv = Convolution::new(client, input, weight, bias, out, args, dtypes);

    match strategy {
        Strategy::Simple {
            read_strategy,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            ReadingStrategy::Cyclic => conv.launch::<SimpleSyncCyclicConv<Accelerated>>(),
            ReadingStrategy::Strided => conv.launch::<SimpleSyncStridedConv<Accelerated>>(),
            ReadingStrategy::Tilewise => conv.launch::<SimpleSyncTilewiseConv<Accelerated>>(),
            ReadingStrategy::AsyncCyclic => conv.launch::<SimpleAsyncCyclicConv<Accelerated>>(),
            ReadingStrategy::AsyncStrided => conv.launch::<SimpleAsyncStridedConv<Accelerated>>(),
            ReadingStrategy::Tma => conv.launch::<SimpleAsyncTmaConv<Accelerated>>(),
        }),
    }
}

#[derive(new)]
struct Convolution<'a, R: Runtime, const N_SPATIAL: usize> {
    client: &'a ComputeClient<R>,
    input: &'a MatmulInputHandleRef<'a, R>,
    weight: &'a MatmulInputHandleRef<'a, R>,
    bias: &'a Option<MatmulInputHandleRef<'a, R>>,
    out: &'a TensorHandleRef<'a, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
}

impl<'a, R: Runtime, const N_SPATIAL: usize> Convolution<'a, R, N_SPATIAL> {
    fn launch<Alg: Algorithm>(self) -> Result<(), ConvSetupError>
    where
        Alg::Args: ConcreteArgs,
    {
        let ConvolutionArgs {
            stride,
            padding,
            dilation,
        } = self.args;

        let dimensionality = match N_SPATIAL {
            1 => Dimensionality::Dim1,
            2 => Dimensionality::Dim2,
            3 => Dimensionality::Dim3,
            other => unimplemented!("Unsupported dimensionality {other}"),
        };

        launch_with_algorithm::<R, Alg>(
            self.client,
            self.input,
            self.weight,
            self.bias,
            self.out,
            (&stride, &padding, &dilation),
            dimensionality,
            self.dtypes,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_with_algorithm<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<MatmulInputHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs,
{
    let rank = input.data().shape.len();
    let dim_c = rank - 1;

    let n = input.data().shape[0];
    let c = input.data().shape[dim_c];

    let out_c = weight.data().shape[0];

    let in_shape = &input.data().shape[1..dim_c];
    let kernel_shape = &weight.data().shape[1..dim_c];
    let out_shape = &out.shape[1..dim_c];

    let op = ConvolutionOperation::Forward;

    let input_data = Alg::into_tensor_handle(client, input.data(), dtypes.lhs_global, op)?;
    let weight_data = Alg::into_tensor_handle(client, weight.data(), dtypes.rhs_global, op)?;

    let mut input = *input;
    let mut weight = *weight;

    *input.data_mut() = input_data.as_ref();
    *weight.data_mut() = weight_data.as_ref();

    let problem = ConvolutionProblem {
        m: n * out_shape.iter().product::<usize>(),
        n: out_c,
        k: c * kernel_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.to_vec(),
        rhs_strides: weight.data().strides.to_vec(),
        lhs_layout: definition::MatrixLayout::RowMajor,
        rhs_layout: definition::MatrixLayout::ColMajor,
        kernel_size: kernel_shape.iter().map(|it| *it as u32).collect(),
        stride: stride.iter().map(|it| *it as u32).collect(),
        padding: padding.iter().map(|it| *it as i32).collect(),
        dilation: dilation.iter().map(|it| *it as u32).collect(),

        batches: n,
        in_shape: in_shape.to_vec(),
        out_shape: out_shape.to_vec(),
        channels: c,
        out_channels: out_c,

        padded_channels: c,
        operation: op,

        dimensionality,
        global_dtypes: dtypes.as_global_elems(),
    };

    launch_kernel::<R, Alg>(client, &input, &weight, bias, out, problem, dtypes)
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<MatmulInputHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    mut dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs,
{
    let plane_dim = client.properties().hardware.plane_size_max;
    // Shape/strides are treated as k-major, with the last dim always being the contiguous one.
    // So for the sake of selecting a line size, the shape/strides are always row-major.
    let line_sizes = AvailableLineSizes::from_type_sizes(
        client,
        input.data().elem_size,
        weight.data().elem_size,
        out.elem_size,
    )
    .filter_lhs_with_tensor(
        input.data().strides,
        input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        weight.data().strides,
        weight.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(out.strides, out.shape);

    let line_sizes = Alg::filter_line_sizes(line_sizes).pick_max()?;

    let selection = Alg::selection(client, &problem, plane_dim, &line_sizes, &mut dtypes)?;
    let problem = Alg::Args::adjust_problem(client, problem, &selection, &dtypes);

    Alg::validate_blueprint(client, &selection, &problem, &dtypes, &line_sizes)?;

    let config = Alg::expand_config(
        client.properties(),
        &problem,
        &selection,
        &line_sizes,
        &dtypes,
    )?;

    let line_sizes = config.line_sizes();

    launch_kernel_concrete::<R, Alg>(
        client, input, weight, bias, out, problem, line_sizes, selection, &dtypes,
    )
}
