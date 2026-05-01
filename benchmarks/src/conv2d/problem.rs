use cubek::convolution::ConvolutionArgs;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_ALEXNET_LIKE: &str = "alexnet_like";
pub const PROBLEM_LARGE_KERNEL: &str = "large_kernel";

pub struct Conv2dProblem {
    pub input_shape: [usize; 4],
    pub weight_shape: [usize; 4],
    pub bias_shape: usize,
    pub args: ConvolutionArgs<2>,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_ALEXNET_LIKE.to_string(),
            label: "AlexNet-like (b=16 in=3x227x227 w=96x3x11x11 s=4)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_LARGE_KERNEL.to_string(),
            label: "Large kernel (b=16 in=4x256x256 w=64x4x8x8)".to_string(),
        },
    ]
}

pub(crate) fn problem_for(id: &str) -> Option<Conv2dProblem> {
    let batch_size = 16;
    Some(match id {
        PROBLEM_ALEXNET_LIKE => Conv2dProblem {
            input_shape: [batch_size, 3, 227, 227],
            weight_shape: [96, 3, 11, 11],
            bias_shape: 96,
            args: ConvolutionArgs {
                stride: [4, 4],
                padding: [0, 0],
                dilation: [1, 1],
            },
        },
        PROBLEM_LARGE_KERNEL => Conv2dProblem {
            input_shape: [batch_size, 4, 256, 256],
            weight_shape: [64, 4, 8, 8],
            bias_shape: 64,
            args: ConvolutionArgs {
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
            },
        },
        _ => return None,
    })
}
