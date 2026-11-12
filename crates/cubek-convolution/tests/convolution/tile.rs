//! The tiled routines against the host's convolution: each operation through each routine, over
//! windows, extents and views that the boxes overhang or fill exactly, and the refusals the launch
//! states before it prepares anything.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, Strides};
use cubek_convolution::components::ConvolutionOperation;
use cubek_convolution::eval::benchmarks::{Conv2dElement, Conv2dProblem, problems};
use cubek_convolution::eval::cpu_reference::{
    HostTensor, naive_data_grad, naive_forward, naive_weight_grad,
};
use cubek_convolution::kernels::tiled::{
    CMMA_STEP_ACROSS_TAPS, CONV_INSTRUCTION_UNOFFERED, CONV_SETTINGS_NOT_TUNED,
    CONV_SETTINGS_NOT_UNIT, CONV_WGMMA_UNSERVED, ConvCandidate, ConvDeclined, ConvRoles,
    ConvRoutineKind, ConvSettings, CubeStagedConvRoutine,
};
use cubek_convolution::{ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::{Hardware, InputBinding};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, StridedLayout, TestInput, TestOutcome,
    assert_equals_approx, launch_and_capture_outcome,
};

use Binding::{Dense, Offset, Permuted, Sliced};
use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// What the sums of the integer operands may differ by: a half rounds nothing below 2048.
const EPS: f32 = 0.01;

/// A value no result holds, which the output starts as so that a cell the kernel leaves alone
/// shows.
const UNTOUCHED: f32 = 100.0;

type Window = ConvolutionArgs<2>;

const fn window(stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2]) -> Window {
    ConvolutionArgs {
        stride,
        padding,
        dilation,
    }
}

const SAME: Window = window([1, 1], [1, 1], [1, 1]);
const VALID: Window = window([1, 1], [0, 0], [1, 1]);
const STRIDE_2: Window = window([2, 2], [1, 1], [1, 1]);
const DILATION_2: Window = window([1, 1], [2, 2], [2, 2]);
const DILATION_UNEQUAL: Window = window([1, 1], [1, 2], [1, 2]);

/// A forward problem of half precision, `channels` in and `filters` out, whose operation a table
/// sets.
fn conv(
    batch: usize,
    (channels, filters): (usize, usize),
    input: [usize; 2],
    kernel: [usize; 2],
    args: Window,
) -> Conv2dProblem {
    Conv2dProblem {
        input_shape: [batch, channels, input[0], input[1]],
        weight_shape: [filters, channels, kernel[0], kernel[1]],
        args,
        operation: Forward,
        element: Conv2dElement::F16,
    }
}

/// [`conv`] over `f32`, which the CPU runtime runs.
fn conv32(
    batch: usize,
    channels: (usize, usize),
    input: [usize; 2],
    kernel: [usize; 2],
    args: Window,
) -> Conv2dProblem {
    Conv2dProblem {
        element: Conv2dElement::F32,
        ..conv(batch, channels, input, kernel, args)
    }
}

/// The catalog's problem `id` over a batch of at most two and a map of at most 27, which keeps its
/// channels, taps, stride and padding.
fn catalog(id: &str) -> Conv2dProblem {
    let entry = problems()
        .into_iter()
        .find(|entry| entry.id == id)
        .unwrap_or_else(|| panic!("unknown problem: {id}"));
    let mut small = entry.value;
    small.input_shape[0] = small.input_shape[0].min(2);
    for side in &mut small.input_shape[2..] {
        *side = (*side).min(27);
    }
    small
}

fn pinned(settings: ConvSettings) -> Strategy {
    Strategy::Tile {
        routine: ConvRoutineKind::CubeStaged,
        settings: Some(settings),
    }
}

/// `settings` over the manual mma instruction.
fn mma(settings: ConvSettings) -> Strategy {
    pinned(settings.mma())
}

/// The unit registers routine, which reads every operand from global memory.
fn unit_registers() -> Strategy {
    Strategy::Tile {
        routine: ConvRoutineKind::UnitRegisters,
        settings: None,
    }
}

/// The setting the autotune finds fastest.
fn tuned() -> Strategy {
    Strategy::Tile {
        routine: ConvRoutineKind::Tuned,
        settings: None,
    }
}

/// How an operand's buffer holds its logical NHWC values.
#[derive(Clone, Copy)]
enum Binding {
    /// Rows of the logical shape, one after another.
    Dense,
    /// The dims in the order of an NCHW tensor, or of the weight's `[kh, kw, ci, co]`.
    Permuted,
    /// Each row of channels stepping past three that are not the view's.
    Sliced,
    /// Dense, a buffer element past its buffer's start.
    Offset,
}

/// How each operand of an operation is bound: its lhs, its rhs and its output.
#[derive(Clone, Copy)]
struct Bindings {
    lhs: Binding,
    rhs: Binding,
    out: Binding,
}

const DENSE: Bindings = Bindings {
    lhs: Dense,
    rhs: Dense,
    out: Dense,
};

const fn lhs(lhs: Binding) -> Bindings {
    Bindings { lhs, ..DENSE }
}

const fn rhs(rhs: Binding) -> Bindings {
    Bindings { rhs, ..DENSE }
}

const fn out(out: Binding) -> Bindings {
    Bindings { out, ..DENSE }
}

const fn lhs_and_rhs(lhs: Binding, rhs: Binding) -> Bindings {
    Bindings { lhs, rhs, ..DENSE }
}

/// Integers a half holds exactly, so that every sum the host makes is exact and the kernel's must
/// match it.
fn values(shape: [usize; 4], salt: usize) -> Vec<f32> {
    (0..shape.iter().product())
        .map(|i| ((i * 5 + salt * 3 + i / 7) % 7) as f32 - 3.0)
        .collect()
}

fn client() -> Client {
    cubecl::test_device().client()
}

fn dtypes(element: ElemType) -> MatmulElems {
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: element,
        rhs: element,
        out: element,
    })
}

/// `values` as a tensor of `shape` bound as `binding` says.
fn operand(
    client: &Client,
    shape: [usize; 4],
    values: Vec<f32>,
    binding: Binding,
    element: ElemType,
) -> TensorHandle {
    let [_, h, w, c] = shape;
    let dense = [h * w * c, w * c, c, 1];
    let strides = match binding {
        Dense | Offset => dense,
        Permuted => [c * h * w, w, 1, h * w],
        Sliced => [h * w * (c + 3), w * (c + 3), c + 3, 1],
    };
    match binding {
        Offset => {
            let padded = [vec![0.0], values].concat();
            let storage = TestInput::builder(client.clone(), [padded.len()])
                .dtype(element)
                .custom(padded)
                .generate_without_host_data();
            let start = storage.handle.offset_start(element.size() as u64);
            TensorHandle::new(start, shape.to_vec(), Strides::new(&strides), element)
        }
        _ => TestInput::builder(client.clone(), shape.to_vec())
            .dtype(element)
            .layout(StridedLayout::Explicit(strides.to_vec()))
            .custom(values)
            .generate_without_host_data(),
    }
}

/// The inputs of `operation`: its lhs and rhs, and the output it writes.
fn inputs(
    operation: ConvolutionOperation,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
) -> ConvolutionInputs {
    match operation {
        Forward => ConvolutionInputs::Forward {
            input: lhs,
            weight: rhs,
            bias: None,
            out,
        },
        BackwardWeight => ConvolutionInputs::BackwardWeight {
            input: lhs,
            out_grad: rhs,
            weight_grad: out,
        },
        BackwardData => ConvolutionInputs::BackwardData {
            out_grad: lhs,
            weights: rhs,
            in_grad: out,
        },
        ConvolutionOperation::ForwardTransposed => {
            unreachable!("no inputs state a transposed forward")
        }
    }
}

fn launch(
    strategy: &Strategy,
    client: &Client,
    inputs: ConvolutionInputs,
    args: Window,
    element: ElemType,
) -> Result<(), String> {
    launch_ref::<2>(strategy, client, inputs, args, dtypes(element)).map_err(|e| format!("{e:?}"))
}

/// The output of `problem`'s operation through `strategy` over operands bound as `bindings` say,
/// and the host's, or the refusal the launch returned.
fn run(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    bindings: Bindings,
) -> Result<(HostData, HostData), String> {
    let client = client();
    let element = problem.element.elem_type();
    let [lhs_shape, rhs_shape, out_shape] = problem.shapes();
    let (lhs_values, rhs_values) = (values(lhs_shape, 2), values(rhs_shape, 4));
    let [_, _, h, w] = problem.input_shape;
    let [_, _, kh, kw] = problem.weight_shape;
    let args = problem.args;
    let lhs_host = HostTensor {
        values: &lhs_values,
        dims: lhs_shape,
    };
    let rhs_host = HostTensor {
        values: &rhs_values,
        dims: rhs_shape,
    };
    let expected = match problem.operation {
        Forward => naive_forward(lhs_host, rhs_host, &args),
        BackwardWeight => naive_weight_grad(lhs_host, rhs_host, [kh, kw], &args),
        BackwardData => naive_data_grad(lhs_host, rhs_host, [h, w], &args),
        ConvolutionOperation::ForwardTransposed => {
            return Err("the host states no transposed forward".to_string());
        }
    };
    let untouched = vec![UNTOUCHED; out_shape.iter().product()];
    let out = operand(&client, out_shape, untouched, bindings.out, element);
    let operands = inputs(
        problem.operation,
        InputBinding::new(
            operand(&client, lhs_shape, lhs_values, bindings.lhs, element).binding(),
            element,
        ),
        InputBinding::new(
            operand(&client, rhs_shape, rhs_values, bindings.rhs, element).binding(),
            element,
        ),
        out.clone().binding(),
    );
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
        match launch(strategy, c, operands, args, element) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(refusal) => ExecutionOutcome::CompileError(refusal),
        }
    });
    if let ExecutionOutcome::CompileError(refusal) = outcome {
        return Err(refusal);
    }
    let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let expected = TestInput::builder(client, out_shape.to_vec())
        .dtype(F32)
        .custom(expected)
        .f32_host_data();
    Ok((actual, expected))
}

/// `problem` as `operation` through `strategy` matches the host's, where the strategy serves it. A
/// device that offers no instruction of the setting's kind is the test mode's to accept; any
/// other refusal fails.
fn check(
    operation: ConvolutionOperation,
    strategy: &Strategy,
    problem: &Conv2dProblem,
    bindings: Bindings,
) {
    let problem = Conv2dProblem {
        operation,
        ..problem.clone()
    };
    match run(strategy, &problem, bindings) {
        Ok((actual, expected)) => assert_equals_approx(&actual, &expected, EPS)
            .as_test_outcome()
            .enforce(),
        Err(refusal) if refusal.contains(why(CONV_INSTRUCTION_UNOFFERED)) => {
            TestOutcome::CompileError(refusal).enforce()
        }
        Err(refusal) => panic!("a setting that serves this problem refused it: {refusal}"),
    }
}

macro_rules! or_dense {
    () => {
        DENSE
    };
    ($bindings:expr) => {
        $bindings
    };
}

/// A module of tests with one test for each row, `name: strategy, problem [=> bindings]`: a launch
/// that faults poisons the device for its process, so no two rows share a test.
macro_rules! cases {
    ($table:ident, $operation:ident; $($name:ident: $strategy:expr, $problem:expr $(=> $bindings:expr)?;)*) => {
        mod $table {
            use super::*;

            $(#[test]
            fn $name() {
                check($operation, &$strategy, &$problem, or_dense!($($bindings)?));
            })*
        }
    };
}

/// A module of modules for the cross product of its settings and its problems: a test for each
/// setting in the module of each problem, `name: strategy` and `name: problem [=> bindings]`.
macro_rules! cross {
    (@problem $operation:ident; $problem:ident: $value:expr, $bindings:expr; [$($setting:ident: $strategy:expr),+]) => {
        mod $problem {
            use super::*;

            $(#[test]
            fn $setting() {
                check($operation, &$strategy, &$value, $bindings);
            })+
        }
    };
    ($table:ident, $operation:ident; settings $settings:tt; problems [$($problem:ident: $value:expr $(=> $bindings:expr)?),+ $(,)?]) => {
        mod $table {
            use super::*;

            $(cross!(@problem $operation; $problem: $value, or_dense!($($bindings)?); $settings);)+
        }
    };
}

/// The cube staged routine, through cmma and through manual mma, which needs plane instructions.
mod cube {
    use super::*;

    cross! {
        forward_whole_channels, Forward;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            cmma_f1x2_p1x2: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f1x2_p1x2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16)
        ];
        problems [
            conv3x3_c64: catalog("conv3x3_c64"),
            conv3x3_c64_stride2: catalog("conv3x3_c64_stride2"),
            narrow_width: catalog("narrow_width"),
        ]
    }

    /// Few channels make a step of taps, which cmma declines by name: only mma serves them.
    cross! {
        forward_taps, Forward;
        settings [
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f1x2_p1x2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16)
        ];
        problems [
            alexnet_like: catalog("alexnet_like"),
            large_kernel: catalog("large_kernel"),
        ]
    }

    /// Vectors read at width one, over whole channels and over a step of taps.
    cases! {
        forward_width_one, Forward;
        conv3x3_c64: mma(CubeStagedConvRoutine::F2X2_P2X2_T16.width_one()), catalog("conv3x3_c64");
        alexnet_like: mma(CubeStagedConvRoutine::F2X2_P2X2_T16.width_one()), catalog("alexnet_like");
    }

    /// Planes across the output rows of a cube, over a map whose rows they overhang, one they
    /// divide, and the 27 rows of the cut catalog problem.
    cross! {
        forward_planes_across, Forward;
        settings [
            cmma_t8: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8),
            cmma_t8_a2: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8.across(2)),
            cmma_t8_a4: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8.across(4)),
            cmma_f1x2_a2: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)),
            cmma_f1x2_a4: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)),
            mma_f1x2_a2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)),
            mma_f1x2_a4: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4))
        ];
        problems [
            narrow_width: catalog("narrow_width"),
            even_rows: conv(2, (64, 64), [8, 7], [3, 3], SAME),
            conv3x3_c64: catalog("conv3x3_c64"),
        ]
    }

    /// Seven planes across a map seven tall, which they divide.
    cases! {
        forward_planes_across_7, Forward;
        cmma_t8_a7: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8.across(7)), catalog("narrow_width");
        cmma_f1x2_a7: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(7)), catalog("narrow_width");
    }

    /// Output channels that are no multiple of a box, so the last box overhangs them, and enough
    /// pixels that a write past them would land in a neighbour's.
    cross! {
        forward_overhanging_filters, Forward;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            filters_3: conv(2, (64, 3), [9, 9], [3, 3], SAME),
            filters_10: conv(2, (64, 10), [20, 20], [3, 3], SAME),
            filters_48: conv(2, (64, 48), [9, 9], [3, 3], SAME),
        ]
    }

    /// An output width that is a whole number of boxes, and ones past 64 that are not.
    cross! {
        forward_output_widths, Forward;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            width_64: conv(1, (64, 32), [3, 64], [3, 3], SAME),
            width_128: conv(1, (64, 32), [3, 128], [3, 3], SAME),
            width_130: conv(1, (64, 32), [3, 130], [3, 3], SAME),
        ]
    }

    cases! {
        forward_output_width_32, Forward;
        cmma_f1x2_p1x2: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16), conv(1, (64, 32), [3, 32], [3, 3], SAME);
    }

    /// Windows that are not the square, dense, symmetric one.
    cross! {
        forward_windows, Forward;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            dilation_2: conv(2, (64, 32), [12, 12], [3, 3], DILATION_2),
            unequal_strides: conv(2, (64, 32), [12, 15], [3, 3], window([1, 2], [1, 1], [1, 1])),
            non_square_kernel: conv(2, (64, 32), [10, 12], [3, 5], window([1, 1], [1, 2], [1, 1])),
            unequal_dilation: conv(2, (64, 32), [12, 15], [3, 3], DILATION_UNEQUAL),
        ]
    }

    /// Settings of 32 rows, at stride 2 and over an overhanging output width.
    cross! {
        forward_rows_32, Forward;
        settings [
            cmma_f2x2_p1x2: pinned(CubeStagedConvRoutine::F2X2_P1X2_T16),
            mma_f1x2_p2x2: mma(CubeStagedConvRoutine::F1X2_P2X2_T16)
        ];
        problems [
            stride_2: conv(2, (64, 32), [12, 15], [3, 3], STRIDE_2),
            width_40: conv(1, (64, 48), [3, 40], [3, 3], SAME),
        ]
    }

    /// Seven instructions of eight rows over a map 56 wide and one it overhangs, and seven rows of
    /// one plane each over maps of seven and nine rows.
    cases! {
        forward_rows_56, Forward;
        rows_56: pinned(CubeStagedConvRoutine::F7X1_P1X2_T8), conv(1, (64, 32), [3, 56], [3, 3], SAME);
        rows_56_overhang: pinned(CubeStagedConvRoutine::F7X1_P1X2_T8), conv(1, (64, 32), [3, 60], [3, 3], SAME);
        one_plane: pinned(CubeStagedConvRoutine::F1X1_P1X1_T8_A7), conv(2, (64, 64), [7, 7], [3, 3], SAME);
        one_plane_overhang: pinned(CubeStagedConvRoutine::F1X1_P1X1_T8_A7), conv(2, (64, 32), [9, 7], [3, 3], SAME);
    }

    /// No padding, where a box that overhangs the output width or the output rows reads past the
    /// input whatever the real positions read: at the last row of the last batch, and across rows.
    cases! {
        forward_no_padding, Forward;
        width_38_cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16), conv(2, (64, 32), [5, 40], [3, 3], VALID);
        width_38_mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16), conv(2, (64, 32), [5, 40], [3, 3], VALID);
        across_4_cmma_t8: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8.across(4)), conv(2, (64, 64), [8, 9], [3, 3], VALID);
        across_4_mma_f1x2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)), conv(2, (64, 64), [8, 9], [3, 3], VALID);
    }

    /// One and five channels, which make a step of taps, and twelve under a padded window: a width
    /// of four divides them, yet a tile of sixteen overhangs them.
    cross! {
        forward_few_channels, Forward;
        settings [mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)];
        problems [
            channels_1: conv(1, (1, 16), [12, 12], [3, 3], SAME),
            channels_5: conv(1, (5, 16), [12, 12], [3, 3], SAME),
            channels_12: conv(2, (12, 16), [9, 9], [3, 3], SAME),
        ]
    }

    cases! {
        forward_channels_12_cmma, Forward;
        cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16), conv(2, (12, 16), [9, 9], [3, 3], SAME);
    }

    /// Views that are not dense rows from the buffer's start: a start a buffer element in, rows
    /// stepping past channels that are not the view's, and a tensor stored channels first.
    cross! {
        forward_bindings, Forward;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            offset_input: conv(2, (64, 32), [6, 7], [3, 3], SAME) => lhs(Offset),
            offset_weight: conv(2, (64, 32), [6, 7], [3, 3], SAME) => rhs(Offset),
            sliced_input: conv(2, (64, 32), [6, 7], [3, 3], SAME) => lhs(Sliced),
            permuted_input: conv(2, (64, 32), [6, 7], [3, 3], SAME) => lhs(Permuted),
            permuted_weight: conv(2, (64, 32), [6, 7], [3, 3], SAME) => rhs(Permuted),
            offset_output: conv(2, (64, 32), [6, 7], [3, 3], SAME) => out(Offset),
        ]
    }

    cases! {
        forward_tuned, Forward;
        narrow_width: tuned(), catalog("narrow_width");
    }

    /// Windows, channels and extents the boxes overhang or fill exactly: output channels and input
    /// channels (the gradient's columns and rows) that are no multiple of a box and that are
    /// whole boxes, and a batch of more than one.
    cross! {
        weight_grad_shapes, BackwardWeight;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            same_3x3_c64: conv(2, (64, 64), [4, 12], [3, 3], SAME),
            stride_2: conv(2, (64, 32), [6, 20], [3, 3], STRIDE_2),
            stride_2_no_padding: conv(2, (32, 32), [7, 21], [3, 3], window([2, 2], [0, 0], [1, 1])),
            dilation_2: conv(2, (64, 32), [8, 17], [3, 3], window([1, 2], [2, 2], [2, 2])),
            dilation_unequal: conv(2, (64, 32), [8, 17], [3, 3], DILATION_UNEQUAL),
            batch_3: conv(3, (32, 16), [4, 12], [3, 3], SAME),
            channels_1: conv(2, (1, 16), [6, 12], [3, 3], SAME),
            channels_3: conv(2, (3, 10), [6, 12], [3, 3], SAME),
            overhang_channels_and_filters: conv(2, (40, 48), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1])),
            whole_channels_128: conv(1, (128, 32), [4, 12], [3, 3], SAME),
            whole_filters_128: conv(1, (32, 128), [4, 12], [3, 3], SAME),
            no_padding: conv(2, (32, 32), [5, 12], [3, 3], VALID),
            kernel_1x1: conv(2, (64, 64), [5, 12], [1, 1], VALID),
            width_21: conv(1, (32, 32), [5, 21], [3, 3], SAME),
        ]
    }

    /// Output maps of one to five columns, which make a step of two axes.
    cross! {
        weight_grad_narrow_maps, BackwardWeight;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            width_3: conv(2, (32, 16), [6, 3], [3, 3], SAME),
            width_1: conv(2, (32, 16), [6, 1], [3, 1], window([1, 1], [1, 0], [1, 1])),
            width_3_stride_2: conv(2, (32, 16), [6, 5], [3, 3], STRIDE_2),
            width_3_dilation_2: conv(2, (32, 16), [6, 5], [3, 3], window([1, 1], [1, 1], [2, 2])),
        ]
    }

    cross! {
        weight_grad_bindings, BackwardWeight;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            offset_views: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs_and_rhs(Offset, Offset),
            sliced_input: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs(Sliced),
            permuted_input: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs(Permuted),
            sliced_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME) => rhs(Sliced),
            permuted_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME) => rhs(Permuted),
            offset_output: conv(2, (64, 32), [5, 12], [3, 3], SAME) => out(Offset),
        ]
    }

    /// Mma at 32 rows over four planes, and with two stages in flight.
    cross! {
        weight_grad_four_planes, BackwardWeight;
        settings [
            mma_f1x2_p2x2: mma(CubeStagedConvRoutine::F1X2_P2X2_T16),
            mma_f1x2_p2x2_double: pinned(CubeStagedConvRoutine::F1X2_P2X2_T16.mma().double_ring())
        ];
        problems [
            stride_1: conv(2, (64, 32), [12, 15], [3, 3], SAME),
            stride_2: conv(2, (64, 32), [12, 15], [3, 3], STRIDE_2),
        ]
    }

    /// Sixteen input channels, whole to an instruction, and planes across the filter's width, with
    /// and without the padding whose overhanging taps gather past the input.
    cross! {
        weight_grad_planes_across, BackwardWeight;
        settings [
            cmma_f1x2_p1x2: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16),
            mma_f1x2_p1x2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16),
            cmma_f1x2_a2: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)),
            cmma_f1x2_a4: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)),
            cmma_t8_a2: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8.across(2)),
            mma_f1x2_a2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)),
            mma_f1x2_a4: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4))
        ];
        problems [
            padded: conv(2, (16, 64), [8, 9], [3, 3], SAME),
            no_padding: conv(2, (16, 64), [8, 9], [3, 3], VALID),
        ]
    }

    /// The setting of 56 rows over a map 56 wide and one it overhangs.
    cases! {
        data_grad_rows_56, BackwardData;
        map_56: pinned(CubeStagedConvRoutine::F7X1_P1X2_T8), conv(1, (64, 32), [3, 56], [3, 3], SAME);
        map_60: pinned(CubeStagedConvRoutine::F7X1_P1X2_T8), conv(1, (64, 32), [3, 60], [3, 3], SAME);
    }

    /// Windows, channels and extents the boxes overhang or fill exactly: input channels (the
    /// gradient's columns) and output channels (the contraction) that are no multiple of a box
    /// and that are whole ones, taps in both directions, and a batch of more than one.
    cross! {
        data_grad_shapes, BackwardData;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            same_3x3_c64: conv(2, (64, 64), [6, 12], [3, 3], SAME),
            no_padding: conv(2, (32, 32), [7, 12], [3, 3], VALID),
            padding_2: conv(2, (32, 32), [7, 12], [3, 3], window([1, 1], [2, 2], [1, 1])),
            dilation_2: conv(2, (64, 32), [8, 17], [3, 3], DILATION_2),
            dilation_unequal: conv(2, (64, 32), [8, 17], [3, 3], DILATION_UNEQUAL),
            overhang_filters_24: conv(2, (32, 24), [6, 12], [3, 3], SAME),
            overhang_filters_40: conv(2, (32, 40), [6, 12], [3, 3], SAME),
            kernel_5x3: conv(2, (32, 32), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1])),
            kernel_1x1: conv(2, (64, 64), [5, 12], [1, 1], VALID),
            batch_3: conv(3, (32, 16), [4, 12], [3, 3], SAME),
            overhang_channels_and_filters: conv(2, (40, 48), [9, 11], [3, 3], SAME),
            channels_1: conv(2, (1, 16), [6, 12], [3, 3], SAME),
            channels_3: conv(2, (3, 16), [6, 12], [3, 3], SAME),
            width_1: conv(2, (32, 32), [6, 1], [3, 1], window([1, 1], [1, 0], [1, 1])),
            width_2: conv(2, (32, 32), [6, 2], [3, 3], SAME),
            width_3: conv(2, (32, 32), [6, 3], [3, 3], SAME),
            width_21: conv(1, (32, 32), [5, 21], [3, 3], SAME),
        ]
    }

    cross! {
        data_grad_bindings, BackwardData;
        settings [
            cmma_f2x2_p2x2: pinned(CubeStagedConvRoutine::F2X2_P2X2_T16),
            mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)
        ];
        problems [
            offset_views: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs_and_rhs(Offset, Offset),
            sliced_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs(Sliced),
            permuted_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs(Permuted),
            sliced_weights: conv(2, (64, 32), [5, 12], [3, 3], SAME) => rhs(Sliced),
            permuted_weights: conv(2, (64, 32), [5, 12], [3, 3], SAME) => rhs(Permuted),
            offset_output: conv(2, (64, 32), [5, 12], [3, 3], SAME) => out(Offset),
        ]
    }

    cases! {
        data_grad_planes_across, BackwardData;
        cmma_f1x2_a2: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
        cmma_f1x2_a4: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
        cmma_t8_a4: pinned(CubeStagedConvRoutine::F1X1_P1X2_T8.across(4)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
        mma_f1x2_a2: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
        mma_f1x2_a4: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
    }

    /// Few output channels make a step of taps and channels, which cmma declines by name.
    cross! {
        data_grad_few_filters, BackwardData;
        settings [mma_f2x2_p2x2: mma(CubeStagedConvRoutine::F2X2_P2X2_T16)];
        problems [
            filters_3: conv(2, (32, 3), [6, 12], [3, 3], SAME),
            filters_8: conv(2, (32, 8), [6, 12], [3, 3], SAME),
            filters_20: conv(2, (32, 20), [6, 12], [3, 3], SAME),
            filters_3_dilation_2: conv(2, (32, 3), [6, 12], [3, 3], DILATION_2),
        ]
    }

    /// The problems every listed setting is held to: a map the boxes overhang, a map seven wide,
    /// and few channels, which make a step of taps.
    fn listed_problems() -> [Conv2dProblem; 3] {
        [
            conv(2, (64, 32), [9, 9], [3, 3], SAME),
            conv(2, (64, 64), [7, 7], [3, 3], SAME),
            conv(1, (3, 16), [9, 9], [3, 3], SAME),
        ]
    }

    /// The refusals a listed setting may be declined with: the ones the routine names, and a stage
    /// past the device's shared memory.
    fn named(message: &str) -> bool {
        [
            CMMA_STEP_ACROSS_TAPS,
            CONV_INSTRUCTION_UNOFFERED,
            CONV_WGMMA_UNSERVED,
        ]
        .into_iter()
        .any(|refusal| message.contains(why(refusal)))
            || message.contains("SharedMemoryPastDevice")
    }

    /// Every setting the device lists, pinned, on a few small problems of `operation`: it matches
    /// the host's result or is declined by name, and some setting serves the operation.
    fn every_listed_setting(operation: ConvolutionOperation) {
        let list = ConvCandidate::list(&Hardware::from_client(&client()));
        assert!(!list.is_empty(), "the device lists no setting");
        let mut served = 0;
        for candidate in &list {
            let strategy = pinned(*candidate.settings());
            for problem in listed_problems() {
                let problem = Conv2dProblem {
                    operation,
                    ..problem
                };
                match run(&strategy, &problem, DENSE) {
                    Ok((actual, expected)) => {
                        assert_equals_approx(&actual, &expected, EPS)
                            .as_test_outcome()
                            .enforce();
                        served += 1;
                    }
                    Err(refusal) => assert!(
                        named(&refusal),
                        "{operation:?} at {} was declined without a name: {refusal}",
                        candidate.settings()
                    ),
                }
            }
        }
        assert!(served > 0, "no listed setting serves {operation:?}");
    }

    #[test]
    fn every_listed_setting_matches_the_forward_or_is_declined_by_name() {
        every_listed_setting(Forward);
    }

    #[test]
    fn every_listed_setting_matches_the_weight_grad_or_is_declined_by_name() {
        every_listed_setting(BackwardWeight);
    }

    #[test]
    fn every_listed_setting_matches_the_data_grad_or_is_declined_by_name() {
        every_listed_setting(BackwardData);
    }

    /// Only a device offering cmma can state this refusal; elsewhere the instruction is refused first.
    #[test]
    fn a_data_grad_step_of_taps_through_cmma_is_refused_by_name() {
        let message = Launch::new(
            BackwardData,
            Plain([2, 6, 6, 3]),
            [3, 3, 3, 32],
            [2, 6, 6, 32],
        )
        .through(pinned(CubeStagedConvRoutine::F2X2_P2X2_T16))
        .refusal();
        assert!(message.contains(why(CMMA_STEP_ACROSS_TAPS)), "{message}");
    }
}

/// The unit registers routine in `f32`, which runs on the CPU runtime: windows and channels the
/// blocks overhang or fill exactly.
mod unit_registers {
    use super::*;

    cases! {
        forward, Forward;
        same_3x3: unit_registers(), conv32(2, (64, 32), [9, 9], [3, 3], SAME);
        stride_2: unit_registers(), conv32(2, (64, 32), [12, 15], [3, 3], STRIDE_2);
        dilation_2: unit_registers(), conv32(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        unequal_strides: unit_registers(), conv32(2, (64, 32), [12, 15], [3, 3], window([1, 2], [1, 1], [1, 1]));
        kernel_1x1: unit_registers(), conv32(2, (64, 32), [5, 12], [1, 1], VALID);
        narrow: unit_registers(), conv32(2, (64, 64), [7, 7], [3, 3], SAME);
        no_padding: unit_registers(), conv32(2, (64, 32), [5, 40], [3, 3], VALID);
        overhanging_channels: unit_registers(), conv32(2, (12, 16), [9, 9], [3, 3], SAME);
        overhanging_filters: unit_registers(), conv32(2, (64, 10), [9, 9], [3, 3], SAME);
        channels_1: unit_registers(), conv32(1, (1, 16), [12, 12], [3, 3], SAME);
        channels_5: unit_registers(), conv32(1, (5, 3), [12, 12], [3, 3], SAME);
        ragged_13x9: unit_registers(), conv32(2, (64, 32), [13, 9], [3, 3], SAME);
        ragged_57x5: unit_registers(), conv32(2, (64, 32), [57, 5], [3, 3], SAME);
    }

    cases! {
        weight_grad, BackwardWeight;
        same_3x3: unit_registers(), conv32(2, (64, 32), [6, 12], [3, 3], SAME);
        stride_2: unit_registers(), conv32(2, (32, 32), [7, 21], [3, 3], STRIDE_2);
        dilation_2: unit_registers(), conv32(2, (64, 32), [8, 17], [3, 3], DILATION_2);
        kernel_1x1: unit_registers(), conv32(2, (64, 64), [5, 12], [1, 1], VALID);
        narrow: unit_registers(), conv32(2, (64, 64), [7, 7], [3, 3], SAME);
        overhanging_channels_and_filters: unit_registers(), conv32(2, (40, 12), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_3: unit_registers(), conv32(2, (3, 10), [6, 12], [3, 3], SAME);
        ragged_13x9: unit_registers(), conv32(2, (64, 32), [13, 9], [3, 3], SAME);
        ragged_57x5: unit_registers(), conv32(2, (64, 32), [57, 5], [3, 3], SAME);
        single_cell: unit_registers(), conv32(1, (1, 1), [3, 3], [1, 1], VALID);
        single_channel_taps: unit_registers(), conv32(1, (1, 1), [5, 5], [3, 3], VALID);
        two_channels_one_filter: unit_registers(), conv32(1, (2, 1), [3, 3], [1, 1], VALID);
        one_channel_two_filters: unit_registers(), conv32(1, (1, 2), [3, 3], [1, 1], VALID);
        permuted_output: unit_registers(), conv32(1, (2, 2), [5, 5], [3, 3], VALID) => out(Permuted);
    }

    cases! {
        data_grad, BackwardData;
        same_3x3: unit_registers(), conv32(2, (64, 32), [6, 12], [3, 3], SAME);
        no_padding: unit_registers(), conv32(2, (32, 32), [7, 12], [3, 3], VALID);
        dilation_2: unit_registers(), conv32(2, (64, 32), [8, 17], [3, 3], DILATION_2);
        kernel_1x1: unit_registers(), conv32(2, (64, 64), [5, 12], [1, 1], VALID);
        narrow: unit_registers(), conv32(2, (64, 64), [7, 7], [3, 3], SAME);
        overhanging_channels_and_filters: unit_registers(), conv32(2, (40, 12), [9, 11], [3, 3], SAME);
        channels_3: unit_registers(), conv32(2, (3, 16), [6, 12], [3, 3], SAME);
        ragged_13x9: unit_registers(), conv32(2, (64, 32), [13, 9], [3, 3], SAME);
        ragged_57x5: unit_registers(), conv32(2, (64, 32), [57, 5], [3, 3], SAME);
    }
}

/// The reason a refusal the routines name states.
fn why(declined: impl Into<Refusal>) -> &'static str {
    match declined.into() {
        Refusal::Declined { why } => why,
        other => panic!("a refusal with no stated reason: {other:?}"),
    }
}

/// The lhs of a launch to refuse, which a quantized one declines whatever else it states.
#[derive(Clone, Copy)]
enum Operand {
    Plain([usize; 4]),
    Quantized([usize; 4]),
}

/// A launch of operands that are dense and of half precision, shaped as stated.
struct Launch {
    operation: ConvolutionOperation,
    strategy: Strategy,
    lhs: Operand,
    rhs: [usize; 4],
    out: [usize; 4],
    args: Window,
}

impl Launch {
    /// The tuned launch of `operation` over the same window.
    fn new(
        operation: ConvolutionOperation,
        lhs: Operand,
        rhs: [usize; 4],
        out: [usize; 4],
    ) -> Self {
        Self {
            operation,
            strategy: tuned(),
            lhs,
            rhs,
            out,
            args: SAME,
        }
    }

    fn over(self, args: Window) -> Self {
        Self { args, ..self }
    }

    fn through(self, strategy: Strategy) -> Self {
        Self { strategy, ..self }
    }

    /// The message the launch is refused with.
    fn refusal(&self) -> String {
        let client = client();
        let dense = |shape: [usize; 4]| {
            let zeros = vec![0.0; shape.iter().product()];
            operand(&client, shape, zeros, Dense, F16).binding()
        };
        let lhs = match self.lhs {
            Operand::Plain(shape) => InputBinding::new(dense(shape), F16),
            Operand::Quantized(shape) => {
                let data = dense(shape);
                InputBinding::Quantized {
                    scale: data.clone(),
                    data,
                    data_dtype: F16,
                    scale_dtype: F32,
                    shape: Shape::from(shape.to_vec()),
                    scheme: Default::default(),
                }
            }
        };
        let operands = inputs(
            self.operation,
            lhs,
            InputBinding::new(dense(self.rhs), F16),
            dense(self.out),
        );
        match launch(&self.strategy, &client, operands, self.args, F16) {
            Ok(()) => panic!("the launch was accepted"),
            Err(refusal) => refusal,
        }
    }
}

/// A test for each launch that must be refused, `name: launch => the refusal it states`. The
/// operands are `[lhs, rhs, out]` in the order of [`Conv2dProblem::shapes`].
macro_rules! refusals {
    ($($name:ident: $launch:expr => $declined:expr;)*) => {
        mod refusals {
            use super::*;

            $(#[test]
            fn $name() {
                let message = $launch.refusal();
                assert!(message.contains(why($declined)), "{message}");
            })*
        }
    };
}

use Operand::{Plain, Quantized};

refusals! {
    forward_quantized_input: Launch::new(Forward, Quantized([1, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 32]) => ConvDeclined::Quantized;
    forward_weight_of_other_channels: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 16], [1, 6, 6, 32]) => ConvDeclined::Groups;
    forward_output_of_other_channels: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 16]) => ConvDeclined::OutChannels;
    forward_output_of_another_batch: Launch::new(Forward, Plain([2, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 32]) => ConvDeclined::Batch;
    forward_output_of_another_extent: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 64], [1, 5, 6, 32]) => ConvDeclined::ForwardExtent;
    forward_weight_of_no_taps: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 0, 3, 64], [1, 6, 6, 32]) => ConvDeclined::KernelEmpty;
    forward_input_of_no_channels: Launch::new(Forward, Plain([1, 6, 6, 0]), [32, 3, 3, 0], [1, 6, 6, 32]) => ConvDeclined::ContractionEmpty;
    forward_stride_of_zero: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 32]).over(window([0, 1], [1, 1], [1, 1])) => ConvDeclined::WindowStepZero;
    forward_dilation_of_zero: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 32]).over(window([1, 1], [1, 1], [1, 0])) => ConvDeclined::WindowStepZero;
    forward_setting_pinned_on_the_autotune: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 32]).through(Strategy::Tile { routine: ConvRoutineKind::Tuned, settings: Some(CubeStagedConvRoutine::F2X2_P2X2_T16) }) => CONV_SETTINGS_NOT_TUNED;
    forward_cube_staged_setting_on_the_unit_routine: Launch::new(Forward, Plain([1, 6, 6, 64]), [32, 3, 3, 64], [1, 6, 6, 32]).through(Strategy::Tile { routine: ConvRoutineKind::UnitRegisters, settings: Some(CubeStagedConvRoutine::F2X2_P2X2_T16) }) => CONV_SETTINGS_NOT_UNIT;

    weight_grad_quantized_input: Launch::new(BackwardWeight, Quantized([1, 6, 6, 64]), [1, 6, 6, 32], [32, 3, 3, 64]) => ConvDeclined::Quantized;
    weight_grad_of_other_channels: Launch::new(BackwardWeight, Plain([1, 6, 6, 64]), [1, 6, 6, 32], [32, 3, 3, 16]) => ConvDeclined::Groups;
    weight_grad_of_other_filters: Launch::new(BackwardWeight, Plain([1, 6, 6, 64]), [1, 6, 6, 32], [16, 3, 3, 64]) => ConvDeclined::OutChannels;
    weight_grad_out_grad_of_another_batch: Launch::new(BackwardWeight, Plain([2, 6, 6, 64]), [1, 6, 6, 32], [32, 3, 3, 64]) => ConvDeclined::Batch;
    weight_grad_out_grad_of_another_extent: Launch::new(BackwardWeight, Plain([1, 6, 6, 64]), [1, 5, 6, 32], [32, 3, 3, 64]) => ConvDeclined::WeightGradExtent;
    weight_grad_of_no_taps: Launch::new(BackwardWeight, Plain([1, 6, 6, 64]), [1, 6, 6, 32], [32, 0, 3, 64]) => ConvDeclined::KernelEmpty;
    weight_grad_over_an_empty_batch: Launch::new(BackwardWeight, Plain([0, 6, 6, 64]), [0, 6, 6, 32], [32, 3, 3, 64]) => ConvDeclined::ContractionEmpty;
    weight_grad_stride_of_zero: Launch::new(BackwardWeight, Plain([1, 6, 6, 64]), [1, 6, 6, 32], [32, 3, 3, 64]).over(window([1, 0], [1, 1], [1, 1])) => ConvDeclined::WindowStepZero;

    data_grad_strided: Launch::new(BackwardData, Plain([1, 3, 3, 32]), [32, 3, 3, 64], [1, 6, 6, 64]).over(STRIDE_2) => ConvDeclined::DataBackwardStrided;
    data_grad_quantized_out_grad: Launch::new(BackwardData, Quantized([1, 6, 6, 32]), [32, 3, 3, 64], [1, 6, 6, 64]) => ConvDeclined::Quantized;
    data_grad_of_other_channels: Launch::new(BackwardData, Plain([1, 6, 6, 32]), [32, 3, 3, 64], [1, 6, 6, 16]) => ConvDeclined::Groups;
    data_grad_of_other_filters: Launch::new(BackwardData, Plain([1, 6, 6, 32]), [16, 3, 3, 64], [1, 6, 6, 64]) => ConvDeclined::OutChannels;
    data_grad_out_grad_of_another_batch: Launch::new(BackwardData, Plain([2, 6, 6, 32]), [32, 3, 3, 64], [1, 6, 6, 64]) => ConvDeclined::Batch;
    data_grad_out_grad_of_another_extent: Launch::new(BackwardData, Plain([1, 5, 6, 32]), [32, 3, 3, 64], [1, 6, 6, 64]) => ConvDeclined::DataGradExtent;
    data_grad_weights_of_no_taps: Launch::new(BackwardData, Plain([1, 6, 6, 32]), [32, 0, 3, 64], [1, 6, 6, 64]) => ConvDeclined::KernelEmpty;
    data_grad_of_no_filters: Launch::new(BackwardData, Plain([1, 6, 6, 0]), [0, 3, 3, 64], [1, 6, 6, 64]) => ConvDeclined::ContractionEmpty;
    data_grad_dilation_of_zero: Launch::new(BackwardData, Plain([1, 6, 6, 32]), [32, 3, 3, 64], [1, 6, 6, 64]).over(window([1, 1], [1, 1], [0, 1])) => ConvDeclined::WindowStepZero;
}

/// A transposed forward is no operation of the tiled convolution, which states it as the data
/// backward at a stride.
#[test]
fn the_transposed_forward_is_refused_by_name() {
    assert_eq!(
        ConvRoles::of(ConvolutionOperation::ForwardTransposed),
        Err(ConvDeclined::Transposed.into())
    );
}

/// An output with no element has nothing to compute: the launch returns and writes nowhere, which
/// the sentinel the output's buffer starts as shows.
#[test]
fn an_empty_output_launches_nothing() {
    let client = client();
    let sentinel = operand(&client, [1, 1, 1, 1], vec![UNTOUCHED], Dense, F16);
    let empty = TensorHandle::new(
        sentinel.handle.clone(),
        vec![1, 0, 0, 32],
        Strides::new(&[1, 1, 1, 1]),
        F16,
    );
    let dense = |shape: [usize; 4]| {
        let ones = vec![1.0; shape.iter().product()];
        InputBinding::new(operand(&client, shape, ones, Dense, F16).binding(), F16)
    };
    let operands = inputs(
        Forward,
        dense([1, 2, 2, 64]),
        dense([32, 3, 3, 64]),
        empty.binding(),
    );
    assert_eq!(launch(&tuned(), &client, operands, VALID, F16), Ok(()));
    let after = HostData::from_tensor_handle(&client, sentinel, HostDataType::F32);
    assert_eq!(after.get_f32(&[0, 0, 0, 0]), UNTOUCHED);
}
