use cubek::convolution::ConvolutionArgs;

use crate::registry::CatalogEntry;

#[derive(Clone)]
pub struct Conv2dProblem {
    pub input_shape: [usize; 4],
    pub weight_shape: [usize; 4],
    pub bias_shape: usize,
    pub args: ConvolutionArgs<2>,
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
            },
        ),
    ]
}
