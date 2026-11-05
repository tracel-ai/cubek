use cubek_test_utils::CatalogEntry;

use crate::{ConvolutionArgs, components::ConvolutionOperation};

#[derive(Clone)]
pub struct Conv2dProblem {
    pub input_shape: [usize; 4],
    pub weight_shape: [usize; 4],
    pub bias_shape: usize,
    pub args: ConvolutionArgs<2>,
    pub operation: ConvolutionOperation,
}

pub fn problems() -> Vec<CatalogEntry<Conv2dProblem>> {
    let batch_size = 16;
    vec![
        CatalogEntry::new(
            "alexnet_like",
            "AlexNet-like (b=16 in=3x227x227 w=96x3x11x11 s=4)",
            Conv2dProblem {
                input_shape: [batch_size, 3, 227, 227],
                weight_shape: [96, 3, 11, 11],
                bias_shape: 96,
                args: ConvolutionArgs {
                    stride: [4, 4],
                    padding: [0, 0],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::Forward,
            },
        ),
        CatalogEntry::new(
            "large_kernel",
            "Large kernel (b=16 in=4x256x256 w=64x4x8x8)",
            Conv2dProblem {
                input_shape: [batch_size, 4, 256, 256],
                weight_shape: [64, 4, 8, 8],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [1, 1],
                    padding: [0, 0],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::Forward,
            },
        ),
        CatalogEntry::new(
            "conv3x3_c64",
            "3x3 stride 1 (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            Conv2dProblem {
                input_shape: [batch_size, 64, 56, 56],
                weight_shape: [64, 64, 3, 3],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [1, 1],
                    padding: [1, 1],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::Forward,
            },
        ),
        CatalogEntry::new(
            "conv3x3_c64_stride2",
            "3x3 stride 2 (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            Conv2dProblem {
                input_shape: [batch_size, 64, 56, 56],
                weight_shape: [64, 64, 3, 3],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [2, 2],
                    padding: [1, 1],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::Forward,
            },
        ),
        CatalogEntry::new(
            "narrow_width",
            "Narrow width (b=64 in=64x7x7 w=64x64x3x3 p=1)",
            Conv2dProblem {
                input_shape: [4 * batch_size, 64, 7, 7],
                weight_shape: [64, 64, 3, 3],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [1, 1],
                    padding: [1, 1],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::Forward,
            },
        ),
        CatalogEntry::new(
            "conv3x3_c64_bwd_data",
            "3x3 stride 1 data backward (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            Conv2dProblem {
                input_shape: [batch_size, 64, 56, 56],
                weight_shape: [64, 64, 3, 3],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [1, 1],
                    padding: [1, 1],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::BackwardData,
            },
        ),
        CatalogEntry::new(
            "conv3x3_c64_bwd_weight",
            "3x3 stride 1 weight backward (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            Conv2dProblem {
                input_shape: [batch_size, 64, 56, 56],
                weight_shape: [64, 64, 3, 3],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [1, 1],
                    padding: [1, 1],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::BackwardWeight,
            },
        ),
        CatalogEntry::new(
            "conv3x3_c64_stride2_bwd_weight",
            "3x3 stride 2 weight backward (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            Conv2dProblem {
                input_shape: [batch_size, 64, 56, 56],
                weight_shape: [64, 64, 3, 3],
                bias_shape: 64,
                args: ConvolutionArgs {
                    stride: [2, 2],
                    padding: [1, 1],
                    dilation: [1, 1],
                },
                operation: ConvolutionOperation::BackwardWeight,
            },
        ),
    ]
}
