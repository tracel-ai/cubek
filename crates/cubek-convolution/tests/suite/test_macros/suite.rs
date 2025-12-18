use crate::suite::convolution_test_launcher::test_convolution_algorithm;
use crate::suite::test_utils::TestPrecision;
use cubecl::Runtime;
use cubecl::frontend::CubePrimitive;
use cubek_convolution::{
    components::{ConvolutionOperation, ConvolutionProblem, Dimensionality},
    forward::args::{ConcreteInputsFactory, ConcreteOutputFactory},
};
use cubek_convolution::{forward::args::ConcreteArgs, kernels::forward::algorithm::Algorithm};
use cubek_matmul::definition::{
    MatmulElemType, MatmulElems, MatmulGlobalElems, MatrixLayout, TilingBlueprint,
};
use cubek_matmul::launch::{InputArg, OutputArg};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionSize {
    pub h: usize,
    pub w: usize,
    pub c: usize,

    pub out_c: usize,
}

pub fn test_algo<A: Algorithm, P: TestPrecision, R: Runtime>(
    blueprint: TilingBlueprint,
    convolution_size: ConvolutionSize,
) where
    A::Args: ConcreteArgs,
{
    let client = R::client(&Default::default());

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

    let elem_type = MatmulElemType {
        dtype: P::EG::as_type_native_unchecked(),
        quantized: false,
    };

    let problem = ConvolutionProblem {
        m: batches * out_h * out_w,
        n: convolution_size.out_c,
        k: kernel_size.iter().product::<u32>() as usize * convolution_size.c,
        lhs_strides: vec![],
        rhs_strides: vec![],
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
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

    test_convolution_algorithm::<A, P, R>(client, problem, blueprint);
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
