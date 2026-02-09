use crate::components::{ConvolutionProblem, Dimensionality};
use crate::{
    AcceleratedTileKind, ReadingStrategy, algorithm::Algorithm,
    components::global::args::RuntimeArgs,
};
use crate::{
    ConvolutionArgs, Strategy, backward_weight::args::ConcreteArgs,
    components::ConvolutionOperation, kernels::algorithm::simple::*,
};
use crate::{
    components::ConvSetupError, kernels::backward_weight::selector::launch_kernel_concrete,
};
use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::*,
    std::{CubeOption, tensor::TensorHandle},
};
use cubek_matmul::launch::{MatmulInputHandle, MatmulInputHandleRef};
use cubek_matmul::{
    components::tile::{cmma::CmmaMatmul, io::Strided, mma::MmaMatmul},
    definition,
};
use cubek_matmul::{
    definition::{AvailableLineSizes, MatmulElems, MatrixLayout},
    routines::BlueprintStrategy,
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
    input: MatmulInputHandle<R>,
    out_grad: MatmulInputHandle<R>,
    weight_grad: TensorHandle<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_ref(
        strategy,
        client,
        &input.as_ref(),
        &out_grad.as_ref(),
        &weight_grad.as_ref(),
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
    out_grad: &MatmulInputHandleRef<'_, R>,
    weight_grad: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let backprop = BackwardsWeight::new(client, input, out_grad, weight_grad, args, dtypes);

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
            ReadingStrategy::Tma => backprop.launch::<SimpleAsyncTmaConv<Accelerated>>(),
        }),
    }
}

#[derive(new)]
struct BackwardsWeight<'a, R: Runtime, const N_SPATIAL: usize> {
    client: &'a ComputeClient<R>,
    input: &'a MatmulInputHandleRef<'a, R>,
    out_grad: &'a MatmulInputHandleRef<'a, R>,
    weight_grad: &'a TensorHandleRef<'a, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
}

impl<'a, R: Runtime, const N_SPATIAL: usize> BackwardsWeight<'a, R, N_SPATIAL> {
    fn launch<Alg: Algorithm>(self) -> Result<(), ConvSetupError>
    where
        Alg::Args: ConcreteArgs<Alg::Routine>,
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
            self.out_grad,
            self.weight_grad,
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
    out_grad: &MatmulInputHandleRef<'_, R>,
    weight_grad: &TensorHandleRef<'_, R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs<Alg::Routine>,
{
    let rank = input.data().shape.len();
    let dim_c = rank - 1;

    let n = input.shape()[0];
    let c = input.shape()[dim_c];

    let out_c = out_grad.shape()[dim_c];

    let in_shape = &input.shape()[1..dim_c];
    let kernel_shape = &weight_grad.shape[1..dim_c];
    let out_shape = &out_grad.shape()[1..dim_c];

    let op = ConvolutionOperation::BackwardWeight;

    let input_data = Alg::into_tensor_handle(client, input.data(), dtypes.lhs_global, op)?;
    let out_grad_data = Alg::into_tensor_handle(client, out_grad.data(), dtypes.rhs_global, op)?;

    let mut input = *input;
    let mut out_grad = *out_grad;

    *input.data_mut() = input_data.as_ref();
    *out_grad.data_mut() = out_grad_data.as_ref();

    let problem = ConvolutionProblem {
        m: out_c,
        n: c * kernel_shape.iter().product::<usize>(),
        k: n * out_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.to_vec(),
        rhs_strides: out_grad.data().strides.to_vec(),
        lhs_layout: definition::MatrixLayout::ColMajor,
        rhs_layout: definition::MatrixLayout::RowMajor,
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

    launch_kernel::<R, Alg>(
        client,
        &input,
        &out_grad,
        weight_grad,
        problem,
        &BlueprintStrategy::Inferred(Default::default()),
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    out_grad: &MatmulInputHandleRef<'_, R>,
    weight_grad: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Alg::Routine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs<Alg::Routine>,
{
    // Shape/strides are treated as k-major, with the last dim always being the contiguous one.
    // So for the sake of selecting a line size, the shape/strides are always row-major.
    let line_sizes = AvailableLineSizes::from_type_sizes(
        client,
        input.data().elem_size,
        out_grad.data().elem_size,
        weight_grad.elem_size,
    )
    .filter_lhs_with_tensor(
        out_grad.data().strides,
        out_grad.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        input.data().strides,
        input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(weight_grad.strides, weight_grad.shape);

    let line_sizes = Alg::filter_line_sizes(line_sizes).pick_max()?;

    launch_kernel_concrete::<R, Alg::Args, Alg::Routine>(
        client,
        input,
        out_grad,
        weight_grad,
        problem,
        line_sizes,
        blueprint_strategy,
        &dtypes,
    )
}
