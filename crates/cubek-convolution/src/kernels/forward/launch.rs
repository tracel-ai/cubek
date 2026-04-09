use crate::{
    AcceleratedTileKind, ReadingStrategy, algorithm::simple::*,
    components::global::args::RuntimeArgs,
};
use crate::{
    ConvolutionArgs, Strategy, components::ConvolutionOperation, forward::args::ConcreteArgs,
};
use crate::{
    algorithm::Algorithm,
    components::{ConvolutionProblem, Dimensionality},
};
use crate::{components::ConvSetupError, kernels::forward::selector::launch_kernel_concrete};
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::{
    components::tile::{cmma::CmmaMatmul, mma::MmaMatmul},
    definition::{AvailableVectorSizes, MatmulElems},
};
use cubek_std::{InputBinding, MatrixLayout};
use derive_new::new;

macro_rules! with_tile_kind {
    ($kind: expr, $T: ident, $launch: expr) => {
        match $kind {
            AcceleratedTileKind::Cmma => {
                type $T = CmmaMatmul;
                ($launch)()
            }
            AcceleratedTileKind::Mma => {
                type $T = MmaMatmul;
                ($launch)()
            }
        }
    };
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
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
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
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
}

impl<'a, R: Runtime, const N_SPATIAL: usize> Convolution<'a, R, N_SPATIAL> {
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
            self.weight,
            self.bias,
            self.out,
            (&stride, &padding, &dilation),
            dimensionality,
            &BlueprintStrategy::Inferred(Default::default()),
            self.dtypes,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_with_algorithm<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    blueprint_strategy: &BlueprintStrategy<RuntimeArgs, Alg::Routine>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Alg::Args: ConcreteArgs<Alg::Routine>,
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

    let input_data = Alg::correct_layout(client, input.clone().into_data(), dtypes.lhs_global, op)?;
    let weight_data =
        Alg::correct_layout(client, weight.clone().into_data(), dtypes.rhs_global, op)?;

    let mut input = input.clone();
    let mut weight = weight.clone();

    *input.data_mut() = input_data;
    *weight.data_mut() = weight_data;

    let address_type = input
        .required_address_type()
        .max(weight.required_address_type())
        .max(
            bias.clone()
                .map(|bias| bias.required_address_type())
                .unwrap_or_default(),
        )
        .max(out.required_address_type(dtypes.acc_global.size()));

    let problem = ConvolutionProblem {
        m: n * out_shape.iter().product::<usize>(),
        n: out_c,
        k: c * kernel_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.clone(),
        rhs_strides: weight.data().strides.clone(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
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

    launch_kernel::<R, Alg>(
        client,
        input,
        weight,
        bias,
        out,
        problem,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
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
        weight.data_elem_size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(
        &input.data().strides,
        &input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        &weight.data().strides,
        &weight.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(&out.strides, &out.shape);

    let mut vector_sizes = Alg::filter_vector_sizes(vector_sizes).pick_max()?;

    // The large vector size resulting from dequantizing ends up slower due to restrictions on
    // algorithms. Use this as a quick and dirty fix.
    if input.scale().is_some() {
        vector_sizes.lhs = 1;
    }
    if weight.scale().is_some() {
        vector_sizes.rhs = 1;
    }

    launch_kernel_concrete::<R, Alg::Args, Alg::Routine>(
        client,
        input,
        weight,
        bias,
        out,
        problem,
        vector_sizes,
        blueprint_strategy,
        &dtypes,
    )
}
