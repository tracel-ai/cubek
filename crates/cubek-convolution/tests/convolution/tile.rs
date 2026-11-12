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

/// A module with a test for each row and each instruction, cmma and manual mma, `name: problem [=>
/// bindings]`, over the 2x2 grid of 16 row tiles.
macro_rules! cube {
    ($table:ident, $operation:ident; $($name:ident: $problem:expr $(=> $bindings:expr)?;)*) => {
        mod $table {
            use super::*;

            $(mod $name {
                use super::*;

                #[test]
                fn through_cmma() {
                    let setting = CubeStagedConvRoutine::F2X2_P2X2_T16;
                    check($operation, &pinned(setting), &$problem, or_dense!($($bindings)?));
                }

                #[test]
                fn through_mma() {
                    let setting = CubeStagedConvRoutine::F2X2_P2X2_T16;
                    check($operation, &mma(setting), &$problem, or_dense!($($bindings)?));
                }
            })*
        }
    };
}

/// The cube staged routine, through cmma and through manual mma, which needs plane instructions.
mod cube {
    use super::*;

    cube! {
        forward, Forward;
        whole_channels: conv(2, (64, 64), [9, 9], [3, 3], SAME);
        stride_2: conv(2, (64, 32), [12, 15], [3, 3], STRIDE_2);
        dilation_2: conv(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        no_padding_overhanging_width: conv(2, (64, 32), [5, 40], [3, 3], VALID);
        overhanging_filters: conv(2, (64, 10), [20, 20], [3, 3], SAME);
        overhanging_channels: conv(2, (12, 16), [9, 9], [3, 3], SAME);
        permuted_weight: conv(2, (64, 32), [6, 7], [3, 3], SAME) => rhs(Permuted);
        offset_output: conv(2, (64, 32), [6, 7], [3, 3], SAME) => out(Offset);
    }

    cube! {
        weight_grad, BackwardWeight;
        whole_channels: conv(2, (64, 64), [4, 12], [3, 3], SAME);
        stride_2: conv(2, (64, 32), [6, 20], [3, 3], STRIDE_2);
        no_padding: conv(2, (32, 32), [5, 12], [3, 3], VALID);
        overhang_channels_and_filters: conv(2, (40, 48), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_3: conv(2, (3, 10), [6, 12], [3, 3], SAME);
        batch_3: conv(3, (32, 16), [4, 12], [3, 3], SAME);
        narrow_map: conv(2, (32, 16), [6, 3], [3, 3], SAME);
        permuted_input: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs(Permuted);
        offset_output: conv(2, (64, 32), [5, 12], [3, 3], SAME) => out(Offset);
    }

    cube! {
        data_grad, BackwardData;
        whole_channels: conv(2, (64, 64), [6, 12], [3, 3], SAME);
        no_padding: conv(2, (32, 32), [7, 12], [3, 3], VALID);
        overhanging_filters: conv(2, (32, 24), [6, 12], [3, 3], SAME);
        channels_3: conv(2, (3, 16), [6, 12], [3, 3], SAME);
        batch_3: conv(3, (32, 16), [4, 12], [3, 3], SAME);
        permuted_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME) => lhs(Permuted);
        offset_output: conv(2, (64, 32), [5, 12], [3, 3], SAME) => out(Offset);
    }

    /// A few channels make a step of taps, which cmma declines by name: only mma serves them.
    cases! {
        mma_taps, Forward;
        channels_1: mma(CubeStagedConvRoutine::F2X2_P2X2_T16), conv(1, (1, 16), [12, 12], [3, 3], SAME);
        large_kernel: mma(CubeStagedConvRoutine::F2X2_P2X2_T16), conv(1, (3, 16), [23, 23], [11, 11], window([4, 4], [1, 1], [1, 1]));
    }

    cases! {
        mma_taps_data_grad, BackwardData;
        filters_3: mma(CubeStagedConvRoutine::F2X2_P2X2_T16), conv(2, (32, 3), [6, 12], [3, 3], SAME);
    }

    /// Planes across the rows of a cube, over a map they overhang, in every direction.
    cases! {
        planes_across_forward, Forward;
        via_cmma: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)), conv(2, (64, 64), [8, 7], [3, 3], SAME);
        via_mma: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)), conv(2, (64, 64), [8, 7], [3, 3], VALID);
    }

    cases! {
        planes_across_weight_grad, BackwardWeight;
        via_cmma: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)), conv(2, (16, 64), [8, 9], [3, 3], SAME);
        via_mma: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)), conv(2, (16, 64), [8, 9], [3, 3], VALID);
    }

    cases! {
        planes_across_data_grad, BackwardData;
        via_cmma: pinned(CubeStagedConvRoutine::F1X2_P1X2_T16.across(2)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
        via_mma: mma(CubeStagedConvRoutine::F1X2_P1X2_T16.across(4)), conv(2, (32, 32), [6, 12], [3, 3], SAME);
    }

    /// Whole steps read at width one, and the 56 row setting over a map it overhangs.
    cases! {
        other_settings, Forward;
        width_one: mma(CubeStagedConvRoutine::F2X2_P2X2_T16.width_one()), conv(2, (64, 64), [9, 9], [3, 3], SAME);
        rows_56_overhang: pinned(CubeStagedConvRoutine::F7X1_P1X2_T8), conv(1, (64, 32), [3, 60], [3, 3], SAME);
    }

    cases! {
        tuned_forward, Forward;
        narrow_map: tuned(), conv(2, (64, 64), [7, 7], [3, 3], SAME);
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

    /// Every setting the device lists, pinned, on one small problem of `operation`: it matches the
    /// host's result or is declined by name, and some setting serves the operation.
    fn every_listed_setting(operation: ConvolutionOperation) {
        let list = ConvCandidate::list(&Hardware::from_client(&client()));
        assert!(!list.is_empty(), "the device lists no setting");
        let problem = Conv2dProblem {
            operation,
            ..conv(2, (64, 32), [9, 9], [3, 3], SAME)
        };
        let mut served = 0;
        for candidate in &list {
            match run(&pinned(*candidate.settings()), &problem, DENSE) {
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
}

/// The unit registers routine in `f32`, which runs on the CPU runtime.
mod unit_registers {
    use super::*;

    cases! {
        forward, Forward;
        same_3x3: unit_registers(), conv32(2, (64, 32), [9, 9], [3, 3], SAME);
        stride_2: unit_registers(), conv32(2, (64, 32), [12, 15], [3, 3], STRIDE_2);
        dilation_2: unit_registers(), conv32(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        no_padding: unit_registers(), conv32(2, (64, 32), [5, 40], [3, 3], VALID);
        overhanging_channels: unit_registers(), conv32(2, (12, 16), [9, 9], [3, 3], SAME);
        overhanging_filters: unit_registers(), conv32(2, (64, 10), [9, 9], [3, 3], SAME);
        channels_1: unit_registers(), conv32(1, (1, 16), [12, 12], [3, 3], SAME);
        ragged_57x5: unit_registers(), conv32(2, (64, 32), [57, 5], [3, 3], SAME);
    }

    cases! {
        weight_grad, BackwardWeight;
        same_3x3: unit_registers(), conv32(2, (64, 32), [6, 12], [3, 3], SAME);
        stride_2: unit_registers(), conv32(2, (32, 32), [7, 21], [3, 3], STRIDE_2);
        overhanging_channels_and_filters: unit_registers(), conv32(2, (40, 12), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_3: unit_registers(), conv32(2, (3, 10), [6, 12], [3, 3], SAME);
        ragged_57x5: unit_registers(), conv32(2, (64, 32), [57, 5], [3, 3], SAME);
        permuted_output: unit_registers(), conv32(1, (2, 2), [5, 5], [3, 3], VALID) => out(Permuted);
    }

    cases! {
        data_grad, BackwardData;
        same_3x3: unit_registers(), conv32(2, (64, 32), [6, 12], [3, 3], SAME);
        no_padding: unit_registers(), conv32(2, (32, 32), [7, 12], [3, 3], VALID);
        dilation_2: unit_registers(), conv32(2, (64, 32), [8, 17], [3, 3], DILATION_2);
        overhanging_channels_and_filters: unit_registers(), conv32(2, (40, 12), [9, 11], [3, 3], SAME);
        channels_3: unit_registers(), conv32(2, (3, 16), [6, 12], [3, 3], SAME);
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

/// A transposed forward is no operation of the tiled convolution, which states it as the data
/// backward at a stride.
#[test]
fn the_transposed_forward_is_refused_by_name() {
    assert_eq!(
        ConvRoles::of(ConvolutionOperation::ForwardTransposed),
        Err(ConvDeclined::Transposed.into())
    );
}
