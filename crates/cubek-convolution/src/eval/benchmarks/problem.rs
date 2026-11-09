use cubecl::{ir::ElemType, prelude::Scalar};
use cubek_test_utils::CatalogEntry;

use crate::{ConvolutionArgs, components::ConvolutionOperation};

/// The element a problem's operands are stored in and its sums run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dElement {
    F16,
    F32,
}

impl Conv2dElement {
    pub fn elem_type(self) -> ElemType {
        match self {
            Self::F16 => half::f16::elem_type_native(),
            Self::F32 => f32::elem_type_native(),
        }
    }
}

#[derive(Clone)]
pub struct Conv2dProblem {
    pub input_shape: [usize; 4],
    pub weight_shape: [usize; 4],
    pub bias_shape: usize,
    pub args: ConvolutionArgs<2>,
    pub operation: ConvolutionOperation,
    pub element: Conv2dElement,
}

impl Conv2dProblem {
    /// The output's height and width.
    pub fn out_extent(&self) -> [usize; 2] {
        let [_, _, h, w] = self.input_shape;
        let [_, _, kh, kw] = self.weight_shape;
        let extent = |side: usize, input: usize, kernel: usize| {
            let args = &self.args;
            (input + 2 * args.padding[side] - args.dilation[side] * (kernel - 1) - 1)
                / args.stride[side]
                + 1
        };
        [extent(0, h, kh), extent(1, w, kw)]
    }
}

pub fn problems() -> Vec<CatalogEntry<Conv2dProblem>> {
    let batch_size = 16;
    let mut problems = vec![
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
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
                element: Conv2dElement::F16,
            },
        ),
    ];
    problems.extend(cpu_problems());
    problems
}

/// One image of f32 through a layer of a small network, for the runtimes that run on a CPU.
fn cpu_problem(
    id: &str,
    (channels_in, channels_out): (usize, usize),
    size: usize,
    kernel: usize,
    stride: usize,
    operation: ConvolutionOperation,
) -> CatalogEntry<Conv2dProblem> {
    let padding = kernel / 2;
    let label = format!(
        "{operation:?} (b=1 in={channels_in}x{size}x{size} w={channels_out}x{channels_in}x{kernel}x{kernel} s={stride} p={padding}, f32)"
    );
    CatalogEntry::new(
        id,
        label,
        Conv2dProblem {
            input_shape: [1, channels_in, size, size],
            weight_shape: [channels_out, channels_in, kernel, kernel],
            bias_shape: channels_out,
            args: ConvolutionArgs {
                stride: [stride; 2],
                padding: [padding; 2],
                dilation: [1, 1],
            },
            operation,
            element: Conv2dElement::F32,
        },
    )
}

fn cpu_problems() -> Vec<CatalogEntry<Conv2dProblem>> {
    use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};
    vec![
        cpu_problem("cpu_3x3_c64_56", (64, 64), 56, 3, 1, Forward),
        cpu_problem("cpu_3x3_s2_c128_56", (128, 128), 56, 3, 2, Forward),
        cpu_problem("cpu_narrow_c64_7", (64, 64), 7, 3, 1, Forward),
        cpu_problem("cpu_1x1_c64_256_56", (64, 256), 56, 1, 1, Forward),
        cpu_problem("cpu_bwd_data_c64_56", (64, 64), 56, 3, 1, BackwardData),
        cpu_problem("cpu_bwd_weight_c64_56", (64, 64), 56, 3, 1, BackwardWeight),
    ]
}
