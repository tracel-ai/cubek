use crate::suite::convolution_test_launcher::test_convolution_algorithm;
use crate::suite::test_utils::TestPrecision;
use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime};
use cubek_convolution::{
    components::{
        ConvolutionOperation, ConvolutionProblem, Dimensionality, global::args::RuntimeArgs,
    },
    forward::args::{ConcreteInputsFactory, ConcreteOutputFactory},
};
use cubek_convolution::{forward::args::ConcreteArgs, kernels::algorithm::Algorithm};
use cubek_matmul::launch::{InputArg, OutputArg};
use cubek_matmul::{
    components::global::{InputLoadFlow, LoadFlows},
    definition::{
        MatmulElems, MatmulGlobalElems, MatrixLayout, SwizzleModes, TilingBlueprint, TilingScheme,
    },
};
use cubek_matmul::{components::stage::PartitionBuffering, routines::Routine};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionSize {
    pub h: usize,
    pub w: usize,
    pub c: usize,

    pub out_c: usize,
}

pub fn test_algo<
    A: Algorithm<Routine: Routine<RuntimeArgs, Blueprint = TilingBlueprint>>,
    P: TestPrecision,
>(
    tiling_scheme: TilingScheme,
    swizzle: SwizzleModes,
    partition_buffering: PartitionBuffering,
    convolution_size: ConvolutionSize,
) where
    A::Args: ConcreteArgs<A::Routine>,
{
    let client = TestRuntime::client(&Default::default());
    let plane_dim = client.properties().hardware.plane_size_max;

    // TODO: Automate more params
    let batches = 2;
    let kernel_size = vec![4, 3];
    let stride = vec![1, 1];
    let padding = vec![3, 1];
    let dilation = vec![3, 2];

    let out_h = calculate_conv_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        convolution_size.h,
    );
    let out_w = calculate_conv_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        convolution_size.w,
    );

    let elem_type = P::EG::as_type_native_unchecked();

    let lhs_layout = MatrixLayout::RowMajor;
    let rhs_layout = MatrixLayout::ColMajor;

    let m = batches * out_h * out_w;
    let k = kernel_size.iter().product::<u32>() as usize * convolution_size.c;
    let n = convolution_size.out_c;

    let lhs_shape = vec![m, k];
    let rhs_shape = vec![k, n];

    let lhs_strides = lhs_layout.to_strides(&lhs_shape);
    let rhs_strides = rhs_layout.to_strides(&rhs_shape);

    let problem = ConvolutionProblem {
        m,
        n,
        k,
        lhs_strides,
        rhs_strides,
        lhs_layout,
        rhs_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        batches,
        in_shape: vec![convolution_size.h, convolution_size.w],
        channels: convolution_size.c,
        out_channels: convolution_size.out_c,
        padded_channels: convolution_size.c,
        out_shape: vec![out_h, out_w],
        dimensionality: Dimensionality::Dim2,
        operation: ConvolutionOperation::Forward,
        global_dtypes: MatmulGlobalElems {
            lhs: elem_type,
            rhs: elem_type,
            out: elem_type,
        },
    };

    let mut blueprint =
        TilingBlueprint::builder(tiling_scheme, plane_dim, &problem.as_matmul_problem())
            .shared_swizzle(swizzle)
            .partition_buffering(partition_buffering);

    if A::IS_SPECIALIZED {
        blueprint = blueprint.load_specialization_config(LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        });
    }

    let blueprint = blueprint.build();

    test_convolution_algorithm::<A, P>(client, problem, blueprint);
}

/// Calculate the expected output size when doing a convolution operation.
pub fn calculate_conv_output_size(
    kernel_size: u32,
    stride: u32,
    padding: i32,
    dilation: u32,
    size_in: usize,
) -> usize {
    (size_in + 2 * padding as usize - dilation as usize * (kernel_size as usize - 1) - 1)
        / stride as usize
        + 1
}
