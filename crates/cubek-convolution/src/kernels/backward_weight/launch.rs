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
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_matmul::components::tile_matmul::TileMatmul;
use cubek_matmul::{
    definition::{AvailableVectorSizes, MatmulElems},
    routines::{BlueprintStrategy, Routine, TilingArgs},
};
use cubek_std::{InputBinding, MatrixLayout};
use derive_new::new;

fn tile_kind_to_dispatch(kind: &AcceleratedTileKind) -> TileMatmul {
    match kind {
        AcceleratedTileKind::Cmma => TileMatmul::Cmma,
        AcceleratedTileKind::Mma => TileMatmul::Mma,
    }
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
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let backprop = BackwardsWeight::new(client, input, out_grad, weight_grad, args, dtypes);

    match strategy {
        Strategy::Simple {
            read_strategy,
            tile_kind,
        } => {
            let kind = tile_kind_to_dispatch(tile_kind);
            match read_strategy {
                ReadingStrategy::Cyclic => backprop.launch::<SimpleSyncCyclicConv>(kind),
                ReadingStrategy::Strided => backprop.launch::<SimpleSyncStridedConv>(kind),
                ReadingStrategy::Tilewise => backprop.launch::<SimpleSyncTilewiseConv>(kind),
                ReadingStrategy::AsyncCyclic => backprop.launch::<SimpleAsyncCyclicConv>(kind),
                ReadingStrategy::AsyncStrided => backprop.launch::<SimpleAsyncStridedConv>(kind),
                ReadingStrategy::Tma => backprop.launch::<SimpleAsyncTmaConv>(kind),
            }
        }
    }
}

#[derive(new)]
struct BackwardsWeight<'a, R: Runtime, const N_SPATIAL: usize> {
    client: &'a ComputeClient<R>,
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
}

impl<'a, R: Runtime, const N_SPATIAL: usize> BackwardsWeight<'a, R, N_SPATIAL> {
    fn launch<Alg: Algorithm>(self, tile_matmul: TileMatmul) -> Result<(), ConvSetupError>
    where
        Alg::Args: ConcreteArgs<Alg::Routine>,
        <Alg::Routine as Routine<RuntimeArgs>>::Strategy: TilingArgs,
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
            tile_matmul,
            self.dtypes,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_with_algorithm<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    tile_matmul: TileMatmul,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs<Alg::Routine>,
    <Alg::Routine as Routine<RuntimeArgs>>::Strategy: TilingArgs,
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

    let input_data = Alg::correct_layout(client, input.clone().into_data(), dtypes.lhs_global, op)?;
    let out_grad_data =
        Alg::correct_layout(client, out_grad.clone().into_data(), dtypes.rhs_global, op)?;

    let mut input = input.clone();
    let mut out_grad = out_grad.clone();

    *input.data_mut() = input_data;
    *out_grad.data_mut() = out_grad_data;

    let address_type = input
        .required_address_type()
        .max(out_grad.required_address_type())
        .max(weight_grad.required_address_type(dtypes.acc_global.size()));

    let problem = ConvolutionProblem {
        m: out_c,
        n: c * kernel_shape.iter().product::<usize>(),
        k: n * out_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.clone(),
        rhs_strides: out_grad.data().strides.clone(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: kernel_shape.iter().map(|it| *it as u32).collect(),
        stride: stride.iter().map(|it| *it as u32).collect(),
        padding: padding.iter().map(|it| *it as i32).collect(),
        dilation: dilation.iter().map(|it| *it as u32).collect(),

        batches: n,
        in_shape: in_shape.into(),
        out_shape: out_shape.into(),
        channels: c,
        out_channels: out_c,

        padded_channels: c,
        operation: op,

        dimensionality,
        global_dtypes: dtypes.as_global_elems(),
        address_type,
    };

    let mut args = <Alg::Routine as Routine<RuntimeArgs>>::Strategy::default();
    args.set_tile_matmul(tile_matmul);

    launch_kernel::<R, Alg>(
        client,
        input,
        out_grad,
        weight_grad,
        problem,
        &BlueprintStrategy::Inferred(args),
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    out_grad: InputBinding<R>,
    weight_grad: TensorBinding<R>,
    problem: ConvolutionProblem,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Alg::Routine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs<Alg::Routine>,
{
    // Shape/strides are treated as k-major, with the last dim always being the contiguous one.
    // So for the sake of selecting a vector size, the shape/strides are always row-major.
    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        input.data_elem_size(),
        out_grad.data_elem_size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(
        &out_grad.data().strides,
        &out_grad.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        &input.data().strides,
        &input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(&weight_grad.strides, &weight_grad.shape);

    let vector_sizes = Alg::filter_vector_sizes(vector_sizes).pick_max()?;

    launch_kernel_concrete::<R, Alg::Args, Alg::Routine>(
        client,
        input,
        out_grad,
        weight_grad,
        problem,
        vector_sizes,
        blueprint_strategy,
        &dtypes,
    )
}
