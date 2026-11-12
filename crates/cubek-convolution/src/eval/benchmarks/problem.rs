use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::Scalar,
};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_test_utils::CatalogEntry;

use crate::{
    ConvolutionArgs,
    components::ConvolutionOperation,
    kernels::tiled::{ConvProblem, ConvRoles},
};

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
    pub args: ConvolutionArgs<2>,
    pub operation: ConvolutionOperation,
    pub element: Conv2dElement,
}

impl Conv2dProblem {
    /// The output's height and width.
    pub fn out_extent(&self) -> [usize; 2] {
        let [_, _, h, w] = self.input_shape;
        let [_, _, kh, kw] = self.weight_shape;
        self.args.output([h, w], [kh, kw])
    }

    /// The shapes of the operands the operation reads and writes, `[lhs, rhs, out]`, each NHWC
    /// but the weight and the weight gradient, which are `[co, kh, kw, ci]`: the input, the weight
    /// and the output of a forward; the input, the output gradient and the weight gradient of a
    /// weight backward; the output gradient, the weight and the input gradient of a data backward.
    pub fn shapes(&self) -> [[usize; 4]; 3] {
        let [n, ci, h, w] = self.input_shape;
        let [co, _, kh, kw] = self.weight_shape;
        let [oh, ow] = self.out_extent();
        let (input, weight, out_grad) = ([n, h, w, ci], [co, kh, kw, ci], [n, oh, ow, co]);
        match self.operation {
            ConvolutionOperation::Forward => [input, weight, out_grad],
            ConvolutionOperation::BackwardWeight => [input, out_grad, weight],
            ConvolutionOperation::BackwardData | ConvolutionOperation::ForwardTransposed => {
                [out_grad, weight, input]
            }
        }
    }
}

/// The problem as the tiled launch states it for dense operands that start at their buffers'
/// starts, summing in `f32`.
impl From<&Conv2dProblem> for ConvProblem {
    fn from(problem: &Conv2dProblem) -> Self {
        let [batch, channels_in, h, w] = problem.input_shape;
        let [channels_out, _, kh, kw] = problem.weight_shape;
        let [oh, ow] = problem.out_extent();
        let roles =
            ConvRoles::of(problem.operation).expect("the host states no transposed forward");
        let [lhs, rhs, out] = problem.shapes();
        let (gathered, out_extent) = match problem.operation {
            ConvolutionOperation::BackwardData | ConvolutionOperation::ForwardTransposed => {
                ([oh, ow], [h, w])
            }
            ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => {
                ([h, w], [oh, ow])
            }
        };
        Self {
            operation: problem.operation,
            batch,
            channels_in,
            gathered,
            channels_out,
            kernel: [kh, kw],
            out: out_extent,
            window: problem.args,
            elements: MatmulOperands::all(problem.element.elem_type()),
            sum: ElemType::Float(FloatKind::F32),
            alignments: roles.dense_alignments(MatmulOperands { lhs, rhs, out }),
            start_bytes: MatmulOperands::all(0),
        }
    }
}

pub fn problems() -> Vec<CatalogEntry<Conv2dProblem>> {
    use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};
    let batch = 16;
    let layer = |stride| ([batch, 64, 56, 56], [64, 64, 3, 3], stride, 1);
    let mut problems = vec![
        half_problem(
            "alexnet_like",
            "AlexNet-like (b=16 in=3x227x227 w=96x3x11x11 s=4)",
            ([batch, 3, 227, 227], [96, 3, 11, 11], 4, 0),
            Forward,
        ),
        half_problem(
            "large_kernel",
            "Large kernel (b=16 in=4x256x256 w=64x4x8x8)",
            ([batch, 4, 256, 256], [64, 4, 8, 8], 1, 0),
            Forward,
        ),
        half_problem(
            "conv3x3_c64",
            "3x3 stride 1 (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            layer(1),
            Forward,
        ),
        half_problem(
            "conv3x3_c64_stride2",
            "3x3 stride 2 (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            layer(2),
            Forward,
        ),
        half_problem(
            "narrow_width",
            "Narrow width (b=64 in=64x7x7 w=64x64x3x3 p=1)",
            ([4 * batch, 64, 7, 7], [64, 64, 3, 3], 1, 1),
            Forward,
        ),
        half_problem(
            "conv3x3_c64_bwd_data",
            "3x3 stride 1 data backward (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            layer(1),
            BackwardData,
        ),
        half_problem(
            "conv3x3_c64_bwd_weight",
            "3x3 stride 1 weight backward (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            layer(1),
            BackwardWeight,
        ),
        half_problem(
            "conv3x3_c64_stride2_bwd_weight",
            "3x3 stride 2 weight backward (b=16 in=64x56x56 w=64x64x3x3 p=1)",
            layer(2),
            BackwardWeight,
        ),
    ];
    problems.extend(cpu_problems());
    problems
}

/// A square-windowed problem of `f16`: `(input, weight, stride, padding)`.
fn half_problem(
    id: &str,
    label: &str,
    (input_shape, weight_shape, stride, padding): ([usize; 4], [usize; 4], usize, usize),
    operation: ConvolutionOperation,
) -> CatalogEntry<Conv2dProblem> {
    CatalogEntry::new(
        id,
        label,
        Conv2dProblem {
            input_shape,
            weight_shape,
            args: ConvolutionArgs {
                stride: [stride; 2],
                padding: [padding; 2],
                dilation: [1, 1],
            },
            operation,
            element: Conv2dElement::F16,
        },
    )
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
