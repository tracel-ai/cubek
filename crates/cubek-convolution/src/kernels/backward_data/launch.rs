use crate::{
    AcceleratedTileKind, ConvolutionArgs, ReadingStrategy, Strategy,
    backward_data::args::ConcreteArgs,
    components::{ConvGemmConfig as _, ConvolutionOperation},
    kernels::forward::simple::*,
};
use crate::{components::ConvSetupError, kernels::backward_data::selector::launch_kernel_concrete};
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
use cubek_matmul::{
    components::tile::{cmma::CmmaMatmul, io::Strided, mma::MmaMatmul},
    definition::{AvailableLineSizes, MatmulElems, MatmulSetupError, MatrixLayout},
    launch::{MatmulInputHandle, MatmulInputHandleRef},
};
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
    out_grad: MatmulInputHandle<R>,
    weights: MatmulInputHandle<R>,
    in_grad: TensorHandle<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_ref(
        strategy,
        client,
        &out_grad.as_ref(),
        &weights.as_ref(),
        &in_grad.as_ref(),
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
    out_grad: &MatmulInputHandleRef<'_, R>,
    weights: &MatmulInputHandleRef<'_, R>,
    in_grad: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let backprop = BackwardsData::new(client, out_grad, weights, in_grad, args, dtypes);

    match strategy {
        Strategy::Simple {
            read_strategy,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            ReadingStrategy::Cyclic => backprop.launch::<SimpleSyncCyclicConv<Accelerated>>(),
            ReadingStrategy::Strided => backprop.launch::<SimpleSyncStridedConv<Accelerated>>(),
            ReadingStrategy::Tilewise => backprop.launch::<SimpleSyncTilewiseConv<Accelerated>>(),
            ReadingStrategy::AsyncCyclic => backprop.launch::<SimpleAsyncCyclicConv<Accelerated>>(),
            ReadingStrategy::AsyncStrided =>
                backprop.launch::<SimpleAsyncStridedConv<Accelerated>>(),
            ReadingStrategy::Tma => Err(ConvSetupError::Matmul(MatmulSetupError::InvalidConfig(
                Box::new("Data backprop doesn't yet work with current TMA tiling strategy")
            ))),
        }),
    }
}

#[derive(new)]
struct BackwardsData<'a, R: Runtime, const N_SPATIAL: usize> {
    client: &'a ComputeClient<R>,
    out_grad: &'a MatmulInputHandleRef<'a, R>,
    weights: &'a MatmulInputHandleRef<'a, R>,
    in_grad: &'a TensorHandleRef<'a, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
}

impl<'a, R: Runtime, const N_SPATIAL: usize> BackwardsData<'a, R, N_SPATIAL> {
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
            self.out_grad,
            self.weights,
            self.in_grad,
            (&stride, &padding, &dilation),
            dimensionality,
            self.dtypes,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_with_algorithm<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    out_grad: &MatmulInputHandleRef<'_, R>,
    weights: &MatmulInputHandleRef<'_, R>,
    in_grad: &TensorHandleRef<'_, R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs,
{
    let rank = in_grad.shape.len();
    let dim_c = rank - 1;

    let n = in_grad.shape[0];
    let c = in_grad.shape[dim_c];

    let out_c = out_grad.shape()[dim_c];

    let in_shape = &in_grad.shape[1..dim_c];
    let kernel_shape = &weights.shape()[1..dim_c];
    let out_shape = &out_grad.shape()[1..dim_c];

    let op = ConvolutionOperation::BackwardData;

    let out_grad_data = Alg::into_tensor_handle(client, out_grad.data(), *dtypes.lhs_global, op)?;
    let weights_data = Alg::into_tensor_handle(client, weights.data(), *dtypes.rhs_global, op)?;

    let mut out_grad = *out_grad;
    let mut weights = *weights;

    *out_grad.data_mut() = out_grad_data.as_ref();
    *weights.data_mut() = weights_data.as_ref();

    let problem = ConvolutionProblem {
        m: n * in_shape.iter().product::<usize>(),
        n: c,
        k: out_c * kernel_shape.iter().product::<usize>(),

        lhs_strides: out_grad.data().strides.to_vec(),
        rhs_strides: weights.data().strides.to_vec(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: kernel_shape.iter().map(|it| *it as u32).collect(),
        stride: stride.iter().map(|it| *it as u32).collect(),
        padding: padding.iter().map(|it| *it as i32).collect(),
        dilation: dilation.iter().map(|it| *it as u32).collect(),

        batches: n,
        in_shape: in_shape.to_vec(),
        out_shape: out_shape.to_vec(),
        channels: c,
        out_channels: out_c,

        padded_channels: out_c,
        operation: op,

        dimensionality,
    };

    launch_kernel::<R, Alg>(client, &out_grad, &weights, in_grad, problem, dtypes)
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    out_grad: &MatmulInputHandleRef<'_, R>,
    weights: &MatmulInputHandleRef<'_, R>,
    in_grad: &TensorHandleRef<'_, R>,
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
        out_grad.data().elem_size,
        weights.data().elem_size,
        in_grad.elem_size,
    )
    .filter_lhs_with_tensor(
        out_grad.data().strides,
        out_grad.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        weights.data().strides,
        weights.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(in_grad.strides, in_grad.shape);

    let line_sizes = Alg::filter_line_sizes(line_sizes).pick_max()?;

    let selection = Alg::selection(client, &problem, plane_dim, &line_sizes, &mut dtypes)?;
    let problem = Alg::Args::adjust_problem(client, problem, &selection, &dtypes);

    let config = Alg::expand_config(client, &problem, &selection, &line_sizes, &dtypes)?;

    let line_sizes = config.line_sizes();

    launch_kernel_concrete::<R, Alg>(
        client, out_grad, weights, in_grad, problem, line_sizes, selection, &dtypes,
    )
}
