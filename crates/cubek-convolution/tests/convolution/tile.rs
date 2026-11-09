//! The tiled routines against the CPU reference, on every conv2d benchmark problem cut down.

#![cfg(feature = "benchmarks")]

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::{Shape, Strides};
use cubek_convolution::eval::benchmarks::{Conv2dCorrectness, Conv2dProblem, problems};
use cubek_convolution::kernels::tiled::{ConvRoutineKind, ConvSettings, CubeStagedConvRoutine};
use cubek_convolution::{
    AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref,
};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::InputBinding;
use cubek_test_utils::{
    Correctness, ExecutionOutcome, HostData, HostDataType, StridedLayout, TestInput, TestOutcome,
    assert_equals_approx, launch_and_capture_outcome,
};

const SEEDS: [u64; 2] = [12, 34];
const F16_EPS: f32 = 0.1;

/// The problem's channels, taps, stride and padding, over a small batch and a small map.
fn small(problem: &Conv2dProblem) -> Conv2dProblem {
    let mut small = problem.clone();
    small.input_shape[0] = small.input_shape[0].min(2);
    for side in &mut small.input_shape[2..] {
        *side = (*side).min(27);
    }
    small
}

use CubeStagedConvRoutine as Staged;

/// The cube staged routine pinned at `settings`.
fn pinned(settings: ConvSettings) -> Strategy {
    Strategy::Tile {
        routine: ConvRoutineKind::CubeStaged,
        settings: Some(settings),
    }
}

/// The cube staged routine at the setting its autotune finds fastest.
fn tuned() -> Strategy {
    Strategy::Tile {
        routine: ConvRoutineKind::CubeStaged,
        settings: None,
    }
}

/// The inferred routine over the manual-mma tile, which the tiled one is held to.
fn legacy_mma() -> Strategy {
    Strategy::Inferred {
        algorithm: ConvAlgorithm::SimpleSyncCyclic,
        tile_kind: AcceleratedTileKind::Mma,
    }
}

/// One test for each line, `check(&strategy, args..)`: a launch that faults poisons the device
/// for its process, so no two cases share a test.
macro_rules! cases {
    ($check:ident; $($name:ident: $strategy:expr $(, $arg:expr)*;)*) => {
        $(#[test]
        fn $name() {
            $check(&$strategy $(, $arg)*);
        })*
    };
}

/// A module for each line, a test in it for the cube routine through cmma and through Mma at 2x2
/// fragments on 2x2 planes, which are the two that serve a step of whole channels.
macro_rules! on_cmma_and_mma {
    ($check:ident; $($stem:ident: $($arg:expr),+;)*) => {
        $(mod $stem {
            use super::*;

            #[test]
            fn cmma_f2x2_p2x2() {
                $check(&pinned(Staged::F2X2_P2X2), $($arg),+);
            }

            #[test]
            fn mma_f2x2_p2x2() {
                $check(&pinned(Staged::F2X2_P2X2.mma()), $($arg),+);
            }
        })*
    };
}

fn check(strategy: &Strategy, problem_id: &str) {
    let entry = problems()
        .into_iter()
        .find(|entry| entry.id == problem_id)
        .unwrap_or_else(|| panic!("unknown problem: {problem_id}"));
    check_problem(strategy, &small(&entry.value));
}

fn check_problem(strategy: &Strategy, problem: &Conv2dProblem) {
    let actual = match Conv2dCorrectness.kernel_result(strategy, problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => panic!("the launch was refused: {e}"),
    };
    let expected = Conv2dCorrectness
        .reference_result(problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed: {e}"));
    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

/// A setting through cmma declines a step made of taps, so it has no test on the two problems with
/// few channels; the first setting that serves them is Mma.
cases! {
    check;
    conv3x3_c64_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), "conv3x3_c64";
    conv3x3_c64_stride2_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), "conv3x3_c64_stride2";
    narrow_width_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), "narrow_width";
    conv3x3_c64_cmma_f1x2_p1x2: pinned(Staged::F1X2_P1X2), "conv3x3_c64";
    conv3x3_c64_stride2_cmma_f1x2_p1x2: pinned(Staged::F1X2_P1X2), "conv3x3_c64_stride2";
    narrow_width_cmma_f1x2_p1x2: pinned(Staged::F1X2_P1X2), "narrow_width";
    conv3x3_c64_cmma_f1x1_p1x2_t8: pinned(Staged::F1X1_P1X2_T8), "conv3x3_c64";
    narrow_width_cmma_f1x1_p1x2_t8: pinned(Staged::F1X1_P1X2_T8), "narrow_width";
    conv3x3_c64_cmma_f1x1_p1x2_t8_a2: pinned(Staged::F1X1_P1X2_T8.across(2)), "conv3x3_c64";
    conv3x3_c64_cmma_f1x1_p1x2_t8_a4: pinned(Staged::F1X1_P1X2_T8.across(4)), "conv3x3_c64";
    conv3x3_c64_cmma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.across(2)), "conv3x3_c64";
    narrow_width_cmma_f1x1_p1x2_t8_a2: pinned(Staged::F1X1_P1X2_T8.across(2)), "narrow_width";
    alexnet_like_mma_f2x2_p2x2_w1: pinned(Staged::F2X2_P2X2.mma().width_one()), "alexnet_like";
    large_kernel_mma_f2x2_p2x2_w1: pinned(Staged::F2X2_P2X2.mma().width_one()), "large_kernel";
    conv3x3_c64_mma_f2x2_p2x2_w1: pinned(Staged::F2X2_P2X2.mma().width_one()), "conv3x3_c64";
    conv3x3_c64_stride2_mma_f2x2_p2x2_w1: pinned(Staged::F2X2_P2X2.mma().width_one()), "conv3x3_c64_stride2";
    narrow_width_mma_f2x2_p2x2_w1: pinned(Staged::F2X2_P2X2.mma().width_one()), "narrow_width";
    alexnet_like_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), "alexnet_like";
    large_kernel_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), "large_kernel";
    conv3x3_c64_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), "conv3x3_c64";
    conv3x3_c64_stride2_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), "conv3x3_c64_stride2";
    narrow_width_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), "narrow_width";
    alexnet_like_mma_f1x2_p1x2: pinned(Staged::F1X2_P1X2.mma()), "alexnet_like";
    large_kernel_mma_f1x2_p1x2: pinned(Staged::F1X2_P1X2.mma()), "large_kernel";
    conv3x3_c64_mma_f1x2_p1x2: pinned(Staged::F1X2_P1X2.mma()), "conv3x3_c64";
    conv3x3_c64_stride2_mma_f1x2_p1x2: pinned(Staged::F1X2_P1X2.mma()), "conv3x3_c64_stride2";
    narrow_width_mma_f1x2_p1x2: pinned(Staged::F1X2_P1X2.mma()), "narrow_width";
    alexnet_like_tuned: tuned(), "alexnet_like";
    large_kernel_tuned: tuned(), "large_kernel";
    conv3x3_c64_tuned: tuned(), "conv3x3_c64";
    conv3x3_c64_stride2_tuned: tuned(), "conv3x3_c64_stride2";
    narrow_width_tuned: tuned(), "narrow_width";
}

/// A forward problem of `channels` in and `channels_out` out, with the window as stated.
fn conv(
    batch: usize,
    (channels, channels_out): (usize, usize),
    input: [usize; 2],
    kernel: [usize; 2],
    window: ([usize; 2], [usize; 2], [usize; 2]),
) -> Conv2dProblem {
    let (stride, padding, dilation) = window;
    Conv2dProblem {
        input_shape: [batch, channels, input[0], input[1]],
        weight_shape: [channels_out, channels, kernel[0], kernel[1]],
        bias_shape: channels_out,
        args: ConvolutionArgs {
            stride,
            padding,
            dilation,
        },
        operation: cubek_convolution::components::ConvolutionOperation::Forward,
    }
}

const SAME_3X3: ([usize; 2], [usize; 2], [usize; 2]) = ([1, 1], [1, 1], [1, 1]);

/// Output channels that are no multiple of a box, so the last box overhangs them.
cases! {
    check_problem;
    co_3_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 3), [9, 9], [3, 3], SAME_3X3);
    co_3_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (64, 3), [9, 9], [3, 3], SAME_3X3);
    co_48_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 48), [9, 9], [3, 3], SAME_3X3);
    co_48_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (64, 48), [9, 9], [3, 3], SAME_3X3);
}

/// Enough pixels that a write past the output channels would land in a neighbour's.
cases! {
    check_problem;
    co_10_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 10), [20, 20], [3, 3], SAME_3X3);
}

/// An output width that is a whole number of boxes, and one past 64 that is not.
cases! {
    check_problem;
    ow_64_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(1, (64, 32), [3, 64], [3, 3], SAME_3X3);
    ow_128_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(1, (64, 32), [3, 128], [3, 3], SAME_3X3);
    ow_32_cmma_f1x2_p1x2: pinned(Staged::F1X2_P1X2), &conv(1, (64, 32), [3, 32], [3, 3], SAME_3X3);
    ow_64_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(1, (64, 32), [3, 64], [3, 3], SAME_3X3);
    ow_130_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(1, (64, 32), [3, 130], [3, 3], SAME_3X3);
}

/// Windows that are not the square, dense, symmetric one.
cases! {
    check_problem;
    dilation_2_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [12, 12], [3, 3], ([1, 1], [2, 2], [2, 2]));
    unequal_strides_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [12, 15], [3, 3], ([1, 2], [1, 1], [1, 1]));
    non_square_kernel_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [10, 12], [3, 5], ([1, 1], [1, 2], [1, 1]));
    unequal_dilation_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [12, 15], [3, 3], ([1, 1], [1, 2], [1, 2]));
    unequal_dilation_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (64, 32), [12, 15], [3, 3], ([1, 1], [1, 2], [1, 2]));
}

/// Settings of 32 rows, at stride 2 and over an overhanging output width.
cases! {
    check_problem;
    stride_2_cmma_f2x2_p1x2: pinned(Staged::F2X2_P1X2), &conv(2, (64, 32), [12, 15], [3, 3], ([2, 2], [1, 1], [1, 1]));
    ow_40_cmma_f2x2_p1x2: pinned(Staged::F2X2_P1X2), &conv(1, (64, 48), [3, 40], [3, 3], SAME_3X3);
    stride_2_mma_f1x2_p2x2: pinned(Staged::F1X2_P2X2.mma()), &conv(2, (64, 32), [12, 15], [3, 3], ([2, 2], [1, 1], [1, 1]));
    ow_40_mma_f1x2_p2x2: pinned(Staged::F1X2_P2X2.mma()), &conv(1, (64, 48), [3, 40], [3, 3], SAME_3X3);
}

/// Seven instructions of eight rows over a map 56 wide, and seven rows of one plane each.
cases! {
    check_problem;
    rows_56_cmma_f7x1_p1x2_t8: pinned(Staged::F7X1_P1X2_T8), &conv(1, (64, 32), [3, 56], [3, 3], SAME_3X3);
    rows_56_overhang_cmma_f7x1_p1x2_t8: pinned(Staged::F7X1_P1X2_T8), &conv(1, (64, 32), [3, 60], [3, 3], SAME_3X3);
    one_plane_cmma_f1x1_p1x1_t8_a7: pinned(Staged::F1X1_P1X1_T8_A7), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    one_plane_overhang_cmma_f1x1_p1x1_t8_a7: pinned(Staged::F1X1_P1X1_T8_A7), &conv(2, (64, 32), [9, 7], [3, 3], SAME_3X3);
}

/// No padding, where a box that overhangs the output width or the output rows reads past the
/// input whatever the real positions read: at the last row of the last batch, and across rows.
cases! {
    check_problem;
    ow_38_no_padding_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [5, 40], [3, 3], ([1, 1], [0, 0], [1, 1]));
    ow_38_no_padding_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (64, 32), [5, 40], [3, 3], ([1, 1], [0, 0], [1, 1]));
    no_padding_cmma_f1x1_p1x2_t8_a4: pinned(Staged::F1X1_P1X2_T8.across(4)), &conv(2, (64, 64), [8, 9], [3, 3], ([1, 1], [0, 0], [1, 1]));
    no_padding_mma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.mma().across(4)), &conv(2, (64, 64), [8, 9], [3, 3], ([1, 1], [0, 0], [1, 1]));
}

/// One and five channels, which make a step of taps, over a batch of one.
cases! {
    check_problem;
    channels_1_tuned: tuned(), &conv(1, (1, 16), [12, 12], [3, 3], SAME_3X3);
    channels_5_tuned: tuned(), &conv(1, (5, 16), [12, 12], [3, 3], SAME_3X3);
}

/// Twelve input channels under a padded window: a width of four divides them, yet a tile of
/// sixteen overhangs them.
cases! {
    check_problem;
    ci_12_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (12, 16), [9, 9], [3, 3], SAME_3X3);
    ci_12_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (12, 16), [9, 9], [3, 3], SAME_3X3);
}

/// Planes across the output rows of a cube, over a map whose rows they overhang, and one they
/// divide.
cases! {
    check_problem;
    narrow_cmma_f1x1_p1x2_t8_a2: pinned(Staged::F1X1_P1X2_T8.across(2)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    narrow_cmma_f1x1_p1x2_t8_a4: pinned(Staged::F1X1_P1X2_T8.across(4)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    narrow_cmma_f1x1_p1x2_t8_a7: pinned(Staged::F1X1_P1X2_T8.across(7)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    narrow_cmma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.across(2)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    narrow_cmma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.across(4)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    narrow_cmma_f1x2_p1x2_a7: pinned(Staged::F1X2_P1X2.across(7)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    even_rows_cmma_f1x1_p1x2_t8_a2: pinned(Staged::F1X1_P1X2_T8.across(2)), &conv(2, (64, 64), [8, 7], [3, 3], SAME_3X3);
    even_rows_cmma_f1x1_p1x2_t8_a4: pinned(Staged::F1X1_P1X2_T8.across(4)), &conv(2, (64, 64), [8, 7], [3, 3], SAME_3X3);
    narrow_mma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.mma().across(2)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
    narrow_mma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.mma().across(4)), &conv(2, (64, 64), [7, 7], [3, 3], SAME_3X3);
}

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// Values a half holds exactly, so the reference sums what the kernel reads.
fn exact_values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + salt) % 13) as f32 / 8.0 - 0.75)
        .collect()
}

/// `input` NHWC and `weight` `[co, kh, kw, ci]`, convolved at stride 1 with `padding`, summed in
/// `f32`.
fn naive(
    (input, [n, h, w, c]): (&[f32], [usize; 4]),
    (weight, [co, kh, kw]): (&[f32], [usize; 3]),
    padding: usize,
) -> (Vec<f32>, [usize; 4]) {
    let (oh, ow) = (h + 2 * padding - kh + 1, w + 2 * padding - kw + 1);
    let mut out = vec![0.0; n * oh * ow * co];
    for b in 0..n {
        for y in 0..oh {
            for x in 0..ow {
                for o in 0..co {
                    let mut sum = 0.0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let (iy, ix) = (y + ky, x + kx);
                            if iy < padding
                                || ix < padding
                                || iy - padding >= h
                                || ix - padding >= w
                            {
                                continue;
                            }
                            for i in 0..c {
                                let at = ((b * h + iy - padding) * w + ix - padding) * c + i;
                                sum += input[at] * weight[((o * kh + ky) * kw + kx) * c + i];
                            }
                        }
                    }
                    out[((b * oh + y) * ow + x) * co + o] = sum;
                }
            }
        }
    }
    (out, [n, oh, ow, co])
}

fn client() -> Client {
    cubecl::test_device().client()
}

fn dtypes() -> MatmulElems {
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: F16,
        rhs: F16,
        out: F16,
    })
}

/// How an operand's buffer holds its logical NHWC values.
#[derive(Clone, Copy)]
enum Binding {
    /// Rows of the logical shape, one after another.
    Dense,
    /// Channels stored in the order of an NCHW tensor.
    Permuted,
    /// Each row of channels stepping past three that are not the view's.
    Sliced,
    /// Dense, a buffer element past its buffer's start.
    Offset,
}

/// `values` as a tensor of `shape` bound as `binding` says.
fn bound(client: &Client, shape: [usize; 4], values: Vec<f32>, binding: Binding) -> TensorBinding {
    let [_, h, w, c] = shape;
    let dense = [h * w * c, w * c, c, 1];
    let strides = match binding {
        Binding::Dense | Binding::Offset => dense,
        Binding::Permuted => [c * h * w, w, 1, h * w],
        Binding::Sliced => [h * w * (c + 3), w * (c + 3), c + 3, 1],
    };
    match binding {
        Binding::Offset => {
            let padded = [vec![0.0], values].concat();
            let flat = TestInput::builder(client.clone(), [padded.len()])
                .dtype(F16)
                .custom(padded)
                .generate_without_host_data();
            let mut view = flat.binding();
            view.shape = Shape::from(shape.to_vec());
            view.strides = Strides::new(&strides);
            view.handle.offset_start = Some(F16.size() as u64);
            view
        }
        _ => TestInput::builder(client.clone(), shape.to_vec())
            .dtype(F16)
            .layout(StridedLayout::Explicit(strides.to_vec()))
            .custom(values)
            .generate_without_host_data()
            .binding(),
    }
}

/// The convolution of a 3x3 weight over a `[2, 6, 7, 64]` input bound as `input` and `weight` say,
/// against the host's.
fn check_bound(strategy: &Strategy, input: Binding, weight: Binding) {
    check_bound_into(strategy, (input, weight, Binding::Dense));
}

/// [`check_bound`] with the output bound as `out` says. It starts as a value no output holds, so
/// a cell the kernel leaves alone shows.
fn check_bound_into(strategy: &Strategy, (input, weight, out): (Binding, Binding, Binding)) {
    let client = client();
    let (shape, filter) = ([2, 6, 7, 64], [32, 3, 3, 64]);
    let (input_values, weight_values) = (
        exact_values(shape.iter().product(), 3),
        exact_values(filter.iter().product(), 5),
    );
    let (expected, out_shape) = naive((&input_values, shape), (&weight_values, [32, 3, 3]), 1);
    let cells: usize = out_shape.iter().product();
    let lead = usize::from(matches!(out, Binding::Offset));
    let storage = TestInput::builder(client.clone(), [lead + cells])
        .dtype(F16)
        .custom(vec![100.0; lead + cells])
        .generate_without_host_data();
    let [_, h, w, c] = out_shape;
    let out = TensorHandle::new(
        storage.handle.offset_start((lead * F16.size()) as u64),
        out_shape.to_vec(),
        Strides::new(&[h * w * c, w * c, c, 1]),
        F16,
    );
    let inputs = ConvolutionInputs::Forward {
        input: InputBinding::new(bound(&client, shape, input_values, input), F16),
        weight: InputBinding::new(bound(&client, filter, weight_values, weight), F16),
        bias: None,
        out: out.clone().binding(),
    };
    let args = ConvolutionArgs {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
        match launch_ref::<2>(strategy, c, inputs, args, dtypes()) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(e) => ExecutionOutcome::CompileError(format!("{e:?}")),
        }
    });
    if let ExecutionOutcome::CompileError(e) = outcome {
        panic!("the launch was refused: {e}");
    }
    let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let expected = TestInput::builder(client, out_shape.to_vec())
        .dtype(F32)
        .custom(expected)
        .f32_host_data();
    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

/// Views that are not dense rows from the buffer's start: a start a buffer element in, rows
/// stepping past channels that are not the view's, and a tensor stored channels first.
cases! {
    check_bound;
    offset_input_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), Binding::Offset, Binding::Dense;
    offset_weight_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), Binding::Dense, Binding::Offset;
    sliced_input_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), Binding::Sliced, Binding::Dense;
    permuted_input_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), Binding::Permuted, Binding::Dense;
    permuted_weight_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), Binding::Dense, Binding::Permuted;
}

cases! {
    check_bound_into;
    offset_output_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), (Binding::Dense, Binding::Dense, Binding::Offset);
    offset_output_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), (Binding::Dense, Binding::Dense, Binding::Offset);
}

/// What the launch of these operands is refused with, as its message.
fn refusal(input: InputBinding, weight: TensorBinding, out: [usize; 4]) -> String {
    let client = client();
    let out = TestInput::builder(client.clone(), out.to_vec())
        .dtype(F16)
        .zeros()
        .generate_without_host_data()
        .binding();
    let inputs = ConvolutionInputs::Forward {
        input,
        weight: InputBinding::new(weight, F16),
        bias: None,
        out,
    };
    let args = ConvolutionArgs {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    match launch_ref::<2>(&tuned(), &client, inputs, args, dtypes()) {
        Ok(()) => panic!("the launch was accepted"),
        Err(e) => format!("{e:?}"),
    }
}

fn dense(client: &Client, shape: [usize; 4]) -> TensorBinding {
    bound(
        client,
        shape,
        exact_values(shape.iter().product(), 1),
        Binding::Dense,
    )
}

#[test]
fn a_quantized_input_is_refused_by_name() {
    let client = client();
    let data = dense(&client, [1, 6, 6, 64]);
    let input = InputBinding::Quantized {
        scale: data.clone(),
        data,
        data_dtype: F16,
        scale_dtype: F32,
        shape: Shape::from(vec![1, 6, 6, 64]),
        scheme: Default::default(),
    };
    let message = refusal(input, dense(&client, [32, 3, 3, 64]), [1, 6, 6, 32]);
    assert!(message.contains("quantized"), "{message}");
}

#[test]
fn an_output_of_other_channels_is_refused_by_name() {
    let client = client();
    let input = InputBinding::new(dense(&client, [1, 6, 6, 64]), F16);
    let message = refusal(input, dense(&client, [32, 3, 3, 64]), [1, 6, 6, 16]);
    assert!(message.contains("output channels"), "{message}");
}

#[test]
fn an_output_of_another_batch_is_refused_by_name() {
    let client = client();
    let input = InputBinding::new(dense(&client, [2, 6, 6, 64]), F16);
    let message = refusal(input, dense(&client, [32, 3, 3, 64]), [1, 6, 6, 32]);
    assert!(message.contains("batch"), "{message}");
}

/// Integers of a small prime's range, which a half holds exactly and no uniform fill can stand in
/// for: every sum the reference makes is exact, so the kernel's must match it.
fn small_integers(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 5 + salt * 3 + i / 7) % 7) as f32 - 3.0)
        .collect()
}

/// `weight_grad[co, rh, rw, ci] = sum over b, oh, ow of input[b, oh*s + rh*d - p, ow*s + rw*d - p,
/// ci] * out_grad[b, oh, ow, co]`, the taps that land outside the input skipped.
fn naive_weight_grad(
    (input, [n, h, w, c]): (&[f32], [usize; 4]),
    (out_grad, [_, oh, ow, co]): (&[f32], [usize; 4]),
    kernel: [usize; 2],
    args: &ConvolutionArgs<2>,
) -> Vec<f32> {
    let mut grad = vec![0.0; co * kernel[0] * kernel[1] * c];
    for o in 0..co {
        for ky in 0..kernel[0] {
            for kx in 0..kernel[1] {
                for i in 0..c {
                    let mut sum = 0.0;
                    for b in 0..n {
                        for y in 0..oh {
                            for x in 0..ow {
                                let at = |out: usize, tap: usize, side: usize| {
                                    (out * args.stride[side] + tap * args.dilation[side]) as isize
                                        - args.padding[side] as isize
                                };
                                let (iy, ix) = (at(y, ky, 0), at(x, kx, 1));
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let input =
                                    input[((b * h + iy as usize) * w + ix as usize) * c + i];
                                sum += input * out_grad[((b * oh + y) * ow + x) * co + o];
                            }
                        }
                    }
                    grad[((o * kernel[0] + ky) * kernel[1] + kx) * c + i] = sum;
                }
            }
        }
    }
    grad
}

/// The weight gradient of `problem` through `strategy`, the input and the output gradient bound
/// as `input` and `out_grad` say, and the host's. The output starts as a value no gradient holds,
/// so a cell the kernel leaves alone shows.
fn weight_grad_values(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    (input, out_grad, weight_grad): (Binding, Binding, Binding),
) -> Result<(HostData, Vec<f32>, [usize; 4]), String> {
    let client = client();
    let [batch, channels, h, w] = problem.input_shape;
    let [filters, _, kh, kw] = problem.weight_shape;
    let args = problem.args.clone();
    let [oh, ow] = problem.out_extent();
    let (input_shape, grad_shape) = ([batch, h, w, channels], [batch, oh, ow, filters]);
    let (input_values, grad_values) = (
        small_integers(input_shape.iter().product(), 1),
        small_integers(grad_shape.iter().product(), 2),
    );
    let expected = naive_weight_grad(
        (&input_values, input_shape),
        (&grad_values, grad_shape),
        [kh, kw],
        &args,
    );
    let weight_shape = [filters, kh, kw, channels];
    let cells: usize = weight_shape.iter().product();
    let lead = usize::from(matches!(weight_grad, Binding::Offset));
    let storage = TestInput::builder(client.clone(), [lead + cells])
        .dtype(F16)
        .custom(vec![100.0; lead + cells])
        .generate_without_host_data();
    let weight_grad = TensorHandle::new(
        storage.handle.offset_start((lead * F16.size()) as u64),
        weight_shape.to_vec(),
        Strides::new(&[kh * kw * channels, kw * channels, channels, 1]),
        F16,
    );
    let inputs = ConvolutionInputs::BackwardWeight {
        input: InputBinding::new(bound(&client, input_shape, input_values, input), F16),
        out_grad: InputBinding::new(bound(&client, grad_shape, grad_values, out_grad), F16),
        weight_grad: weight_grad.clone().binding(),
    };
    let outcome =
        launch_and_capture_outcome(&client, &[&weight_grad.handle], |c| {
            match launch_ref::<2>(strategy, c, inputs, args, dtypes()) {
                Ok(()) => ExecutionOutcome::Executed,
                Err(e) => ExecutionOutcome::CompileError(format!("{e:?}")),
            }
        });
    if let ExecutionOutcome::CompileError(e) = outcome {
        return Err(e);
    }
    let actual = HostData::from_tensor_handle(&client, weight_grad, HostDataType::F32);
    Ok((actual, expected, weight_shape))
}

fn check_weight_grad(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    input: Binding,
    out_grad: Binding,
) {
    check_weight_grad_into(strategy, problem, (input, out_grad, Binding::Dense));
}

fn check_weight_grad_into(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    bindings: (Binding, Binding, Binding),
) {
    let (actual, expected, shape) = match weight_grad_values(strategy, problem, bindings) {
        Ok(values) => values,
        Err(e) => panic!("a setting that serves this problem refused it: {e}"),
    };
    let expected = TestInput::builder(client(), shape.to_vec())
        .dtype(F32)
        .custom(expected)
        .f32_host_data();
    assert_equals_approx(&actual, &expected, GRAD_EPS)
        .as_test_outcome()
        .enforce();
}

/// What the sums of small integers may differ by: the half the output is stored in rounds nothing
/// below 2048.
const GRAD_EPS: f32 = 0.01;

/// Windows, channels and extents the boxes overhang or fill exactly: output channels and input
/// channels (the gradient's columns and rows) that are no multiple of a box and that are whole
/// boxes, and a batch of more than one.
on_cmma_and_mma! {
    check_weight_grad;
    weight_grad_same_3x3_c64: &conv(2, (64, 64), [4, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_stride_2: &conv(2, (64, 32), [6, 20], [3, 3], ([2, 2], [1, 1], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_stride_2_no_padding: &conv(2, (32, 32), [7, 21], [3, 3], ([2, 2], [0, 0], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_dilation_2: &conv(2, (64, 32), [8, 17], [3, 3], ([1, 2], [2, 2], [2, 2])), Binding::Dense, Binding::Dense;
    weight_grad_unequal_dilation: &conv(2, (64, 32), [8, 17], [3, 3], ([1, 1], [1, 2], [1, 2])), Binding::Dense, Binding::Dense;
    weight_grad_batch_3: &conv(3, (32, 16), [4, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_channels_1: &conv(2, (1, 16), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_channels_3: &conv(2, (3, 10), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_overhang_ci_co: &conv(2, (40, 48), [9, 11], [5, 3], ([1, 1], [2, 1], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_whole_ci_128_co_32: &conv(1, (128, 32), [4, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_whole_ci_32_co_128: &conv(1, (32, 128), [4, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_no_padding: &conv(2, (32, 32), [5, 12], [3, 3], ([1, 1], [0, 0], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_1x1: &conv(2, (64, 64), [5, 12], [1, 1], ([1, 1], [0, 0], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_ow_21: &conv(1, (32, 32), [5, 21], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_offset_views: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Offset, Binding::Offset;
    weight_grad_sliced_input: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Sliced, Binding::Dense;
    weight_grad_permuted_out_grad: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Permuted;
    weight_grad_permuted_input: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Permuted, Binding::Dense;
    weight_grad_sliced_out_grad: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Sliced;
}

cases! {
    check_weight_grad_into;
    weight_grad_offset_output_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), (Binding::Dense, Binding::Dense, Binding::Offset);
    weight_grad_offset_output_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), (Binding::Dense, Binding::Dense, Binding::Offset);
}

/// The Mma setting of 32 rows over four planes, at stride 1 and 2.
#[test]
fn weight_grad_cube_staged_mma_grid_1x2_on_2x2() {
    check_weight_grad(
        &pinned(Staged::F1X2_P2X2.mma()),
        &conv(2, (64, 32), [12, 15], [3, 3], SAME_3X3),
        Binding::Dense,
        Binding::Dense,
    );
    check_weight_grad(
        &pinned(Staged::F1X2_P2X2.mma()),
        &conv(2, (64, 32), [12, 15], [3, 3], ([2, 2], [1, 1], [1, 1])),
        Binding::Dense,
        Binding::Dense,
    );
}

/// The same setting with two stages in flight, at stride 2.
#[test]
fn weight_grad_cube_staged_mma_grid_1x2_on_2x2_double() {
    check_weight_grad(
        &pinned(Staged::F1X2_P2X2.mma().double_ring()),
        &conv(2, (64, 32), [12, 15], [3, 3], ([2, 2], [1, 1], [1, 1])),
        Binding::Dense,
        Binding::Dense,
    );
}

cases! {
    check_weight_grad;
    weight_grad_ow_3_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 16), [6, 3], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_ow_1_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 16), [6, 1], [3, 1], ([1, 1], [1, 0], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_ow_3_stride_2_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 16), [6, 5], [3, 3], ([2, 2], [1, 1], [1, 1])), Binding::Dense, Binding::Dense;
    weight_grad_ow_3_dilation_2_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 16), [6, 5], [3, 3], ([1, 1], [1, 1], [2, 2])), Binding::Dense, Binding::Dense;
}

cases! {
    check_weight_grad;
    two_axes_weight_grad_ow_3_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (32, 16), [6, 3], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    two_axes_weight_grad_ow_1_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (32, 16), [6, 1], [3, 1], ([1, 1], [1, 0], [1, 1])), Binding::Dense, Binding::Dense;
    two_axes_weight_grad_ow_3_stride_2_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (32, 16), [6, 5], [3, 3], ([2, 2], [1, 1], [1, 1])), Binding::Dense, Binding::Dense;
    two_axes_weight_grad_ow_3_dilation_2_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (32, 16), [6, 5], [3, 3], ([1, 1], [1, 1], [2, 2])), Binding::Dense, Binding::Dense;
}

cases! {
    check_weight_grad;
    weight_grad_whole_ci_16_cmma_f1x2_p1x2: pinned(Staged::F1X2_P1X2), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_whole_ci_16_mma_f1x2_p1x2: pinned(Staged::F1X2_P1X2.mma()), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
}

cases! {
    check_weight_grad;
    weight_grad_cmma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.across(2)), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_cmma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.across(4)), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_cmma_f1x1_p1x2_t8_a2: pinned(Staged::F1X1_P1X2_T8.across(2)), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_mma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.mma().across(2)), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    weight_grad_mma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.mma().across(4)), &conv(2, (16, 64), [8, 9], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
}

/// Planes across the filter's width with no padding, whose overhanging taps gather past the input.
#[test]
fn weight_grad_across_2_no_padding_cube_staged_cmma_grid_1x2() {
    let problem = conv(2, (16, 64), [8, 9], [3, 3], ([1, 1], [0, 0], [1, 1]));
    check_weight_grad(
        &pinned(Staged::F1X2_P1X2.across(2)),
        &problem,
        Binding::Dense,
        Binding::Dense,
    );
}

/// What the launch of these weight backward operands through `strategy` is refused with, as its
/// message.
fn weight_grad_refusal(
    strategy: &Strategy,
    input: InputBinding,
    out_grad: TensorBinding,
    weight_grad: [usize; 4],
) -> String {
    let client = client();
    let weight_grad = TestInput::builder(client.clone(), weight_grad.to_vec())
        .dtype(F16)
        .zeros()
        .generate_without_host_data()
        .binding();
    let inputs = ConvolutionInputs::BackwardWeight {
        input,
        out_grad: InputBinding::new(out_grad, F16),
        weight_grad,
    };
    let args = ConvolutionArgs {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    match launch_ref::<2>(strategy, &client, inputs, args, dtypes()) {
        Ok(()) => panic!("the launch was accepted"),
        Err(e) => format!("{e:?}"),
    }
}

#[test]
fn a_quantized_weight_backward_input_is_refused_by_name() {
    let client = client();
    let data = dense(&client, [1, 6, 6, 64]);
    let input = InputBinding::Quantized {
        scale: data.clone(),
        data,
        data_dtype: F16,
        scale_dtype: F32,
        shape: Shape::from(vec![1, 6, 6, 64]),
        scheme: Default::default(),
    };
    let out_grad = dense(&client, [1, 6, 6, 32]);
    let message = weight_grad_refusal(&tuned(), input, out_grad, [32, 3, 3, 64]);
    assert!(message.contains("quantized"), "{message}");
}

#[test]
fn a_weight_gradient_of_other_channels_is_refused_by_name() {
    let client = client();
    let input = InputBinding::new(dense(&client, [1, 6, 6, 64]), F16);
    let out_grad = dense(&client, [1, 6, 6, 32]);
    let message = weight_grad_refusal(&tuned(), input, out_grad, [32, 3, 3, 16]);
    assert!(message.contains("no groups"), "{message}");
}

#[test]
fn a_weight_gradient_of_other_filters_is_refused_by_name() {
    let client = client();
    let input = InputBinding::new(dense(&client, [1, 6, 6, 64]), F16);
    let out_grad = dense(&client, [1, 6, 6, 32]);
    let message = weight_grad_refusal(&tuned(), input, out_grad, [16, 3, 3, 64]);
    assert!(message.contains("output channels"), "{message}");
}

#[test]
fn an_output_gradient_of_another_batch_is_refused_by_name() {
    let client = client();
    let input = InputBinding::new(dense(&client, [2, 6, 6, 64]), F16);
    let out_grad = dense(&client, [1, 6, 6, 32]);
    let message = weight_grad_refusal(&tuned(), input, out_grad, [32, 3, 3, 64]);
    assert!(message.contains("batch"), "{message}");
}

/// `in_grad[b, y, x, ci] = sum over co, ky, kx of out_grad[b, gy, gx, co] * weights[co, ky, kx, ci]`
/// where `g = y + padding - tap * dilation`, the positions outside the gradient skipped.
fn naive_data_grad(
    (out_grad, [n, oh, ow, co]): (&[f32], [usize; 4]),
    (weights, [_, kh, kw, c]): (&[f32], [usize; 4]),
    [h, w]: [usize; 2],
    args: &ConvolutionArgs<2>,
) -> Vec<f32> {
    let mut grad = vec![0.0; n * h * w * c];
    for b in 0..n {
        for y in 0..h {
            for x in 0..w {
                for i in 0..c {
                    let mut sum = 0.0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let at = |pos: usize, tap: usize, side: usize| {
                                (pos + args.padding[side]) as isize
                                    - (tap * args.dilation[side]) as isize
                            };
                            let (gy, gx) = (at(y, ky, 0), at(x, kx, 1));
                            if gy < 0 || gx < 0 || gy >= oh as isize || gx >= ow as isize {
                                continue;
                            }
                            for o in 0..co {
                                let g =
                                    out_grad[((b * oh + gy as usize) * ow + gx as usize) * co + o];
                                sum += g * weights[((o * kh + ky) * kw + kx) * c + i];
                            }
                        }
                    }
                    grad[((b * h + y) * w + x) * c + i] = sum;
                }
            }
        }
    }
    grad
}

/// The data gradient of `problem` (read as the forward one it is the gradient of) through
/// `strategy`, the output gradient and the weights bound as `out_grad` and `weights` say, and
/// the host's. The output starts as a value no gradient holds, so a cell the kernel leaves alone
/// shows.
fn data_grad_values(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    (out_grad, weights, in_grad): (Binding, Binding, Binding),
) -> Result<(HostData, Vec<f32>, [usize; 4]), String> {
    let client = client();
    let [batch, channels, h, w] = problem.input_shape;
    let [filters, _, kh, kw] = problem.weight_shape;
    let args = problem.args.clone();
    let [oh, ow] = problem.out_extent();
    let (grad_shape, weight_shape) = ([batch, oh, ow, filters], [filters, kh, kw, channels]);
    let (grad_values, weight_values) = (
        small_integers(grad_shape.iter().product(), 2),
        small_integers(weight_shape.iter().product(), 4),
    );
    let expected = naive_data_grad(
        (&grad_values, grad_shape),
        (&weight_values, weight_shape),
        [h, w],
        &args,
    );
    let in_shape = [batch, h, w, channels];
    let cells: usize = in_shape.iter().product();
    let lead = usize::from(matches!(in_grad, Binding::Offset));
    let storage = TestInput::builder(client.clone(), [lead + cells])
        .dtype(F16)
        .custom(vec![100.0; lead + cells])
        .generate_without_host_data();
    let in_grad = TensorHandle::new(
        storage.handle.offset_start((lead * F16.size()) as u64),
        in_shape.to_vec(),
        Strides::new(&[h * w * channels, w * channels, channels, 1]),
        F16,
    );
    let inputs = ConvolutionInputs::BackwardData {
        out_grad: InputBinding::new(bound(&client, grad_shape, grad_values, out_grad), F16),
        weights: InputBinding::new(bound(&client, weight_shape, weight_values, weights), F16),
        in_grad: in_grad.clone().binding(),
    };
    let outcome = launch_and_capture_outcome(&client, &[&in_grad.handle], |c| {
        match launch_ref::<2>(strategy, c, inputs, args, dtypes()) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(e) => ExecutionOutcome::CompileError(format!("{e:?}")),
        }
    });
    if let ExecutionOutcome::CompileError(e) = outcome {
        return Err(e);
    }
    let actual = HostData::from_tensor_handle(&client, in_grad, HostDataType::F32);
    Ok((actual, expected, in_shape))
}

fn check_data_grad_into(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    bindings: (Binding, Binding, Binding),
) {
    let (actual, expected, shape) = match data_grad_values(strategy, problem, bindings) {
        Ok(values) => values,
        Err(e) => panic!("a setting that serves this problem refused it: {e}"),
    };
    let expected = TestInput::builder(client(), shape.to_vec())
        .dtype(F32)
        .custom(expected)
        .f32_host_data();
    assert_equals_approx(&actual, &expected, GRAD_EPS)
        .as_test_outcome()
        .enforce();
}

fn check_data_grad(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    out_grad: Binding,
    weights: Binding,
) {
    check_data_grad_into(strategy, problem, (out_grad, weights, Binding::Dense));
}

/// The setting of 56 rows over a map 56 wide and one it overhangs.
#[test]
fn data_grad_cube_staged_cmma_56_rows() {
    for problem in [
        conv(1, (64, 32), [3, 56], [3, 3], SAME_3X3),
        conv(1, (64, 32), [3, 60], [3, 3], SAME_3X3),
    ] {
        check_data_grad(
            &pinned(Staged::F7X1_P1X2_T8),
            &problem,
            Binding::Dense,
            Binding::Dense,
        );
    }
}

/// Windows, channels and extents the boxes overhang or fill exactly: input channels (the gradient's
/// columns) and output channels (the contraction) that are no multiple of a box and that are whole
/// ones, taps in both directions, and a batch of more than one.
on_cmma_and_mma! {
    check_data_grad;
    data_grad_same_3x3_c64: &conv(2, (64, 64), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_no_padding: &conv(2, (32, 32), [7, 12], [3, 3], ([1, 1], [0, 0], [1, 1])), Binding::Dense, Binding::Dense;
    data_grad_padding_2: &conv(2, (32, 32), [7, 12], [3, 3], ([1, 1], [2, 2], [1, 1])), Binding::Dense, Binding::Dense;
    data_grad_dilation_2: &conv(2, (64, 32), [8, 17], [3, 3], ([1, 1], [2, 2], [2, 2])), Binding::Dense, Binding::Dense;
    data_grad_unequal_dilation: &conv(2, (64, 32), [8, 17], [3, 3], ([1, 1], [1, 2], [1, 2])), Binding::Dense, Binding::Dense;
    data_grad_overhang_co_24: &conv(2, (32, 24), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_overhang_co_40: &conv(2, (32, 40), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_kernel_5x3: &conv(2, (32, 32), [9, 11], [5, 3], ([1, 1], [2, 1], [1, 1])), Binding::Dense, Binding::Dense;
    data_grad_1x1: &conv(2, (64, 64), [5, 12], [1, 1], ([1, 1], [0, 0], [1, 1])), Binding::Dense, Binding::Dense;
    data_grad_batch_3: &conv(3, (32, 16), [4, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_overhang_ci_co: &conv(2, (40, 48), [9, 11], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_channels_1: &conv(2, (1, 16), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_channels_3: &conv(2, (3, 16), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_width_1: &conv(2, (32, 32), [6, 1], [3, 1], ([1, 1], [1, 0], [1, 1])), Binding::Dense, Binding::Dense;
    data_grad_width_2: &conv(2, (32, 32), [6, 2], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_width_3: &conv(2, (32, 32), [6, 3], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_width_21: &conv(1, (32, 32), [5, 21], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_offset_views: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Offset, Binding::Offset;
    data_grad_sliced_out_grad: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Sliced, Binding::Dense;
    data_grad_sliced_weights: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Sliced;
    data_grad_permuted_out_grad: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Permuted, Binding::Dense;
    data_grad_permuted_weights: &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Permuted;
}

cases! {
    check_data_grad;
    data_grad_cmma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.across(2)), &conv(2, (32, 32), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_cmma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.across(4)), &conv(2, (32, 32), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_cmma_f1x1_p1x2_t8_a4: pinned(Staged::F1X1_P1X2_T8.across(4)), &conv(2, (32, 32), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_mma_f1x2_p1x2_a2: pinned(Staged::F1X2_P1X2.mma().across(2)), &conv(2, (32, 32), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_mma_f1x2_p1x2_a4: pinned(Staged::F1X2_P1X2.mma().across(4)), &conv(2, (32, 32), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
}

cases! {
    check_data_grad_into;
    data_grad_offset_output_cmma_f2x2_p2x2: pinned(Staged::F2X2_P2X2), &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), (Binding::Dense, Binding::Dense, Binding::Offset);
    data_grad_offset_output_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (64, 32), [5, 12], [3, 3], SAME_3X3), (Binding::Dense, Binding::Dense, Binding::Offset);
}

cases! {
    check_data_grad;
    data_grad_co_3_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 3), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_co_8_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 8), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_overhang_co_20_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 20), [6, 12], [3, 3], SAME_3X3), Binding::Dense, Binding::Dense;
    data_grad_co_3_dilation_2_mma_f2x2_p2x2: pinned(Staged::F2X2_P2X2.mma()), &conv(2, (32, 3), [6, 12], [3, 3], ([1, 1], [2, 2], [2, 2])), Binding::Dense, Binding::Dense;
}

/// The data gradient of one shape through the legacy Mma kernel and through the tiled one, which
/// must agree at every element.
#[test]
fn data_grad_matches_the_legacy_kernel_on_mma() {
    let problem = conv(2, (64, 32), [6, 12], [3, 3], SAME_3X3);
    let bindings = (Binding::Dense, Binding::Dense, Binding::Dense);
    let legacy = data_grad_values(&legacy_mma(), &problem, bindings)
        .unwrap_or_else(|e| panic!("the legacy launch was refused: {e}"))
        .0;
    let tiled = data_grad_values(&pinned(Staged::F2X2_P2X2.mma()), &problem, bindings)
        .unwrap_or_else(|e| panic!("the tiled launch was refused: {e}"))
        .0;
    assert_equals_approx(&tiled, &legacy, GRAD_EPS)
        .as_test_outcome()
        .enforce();
}

/// What the launch of these data backward operands through `strategy` over `args` is refused
/// with, as its message.
fn data_grad_refusal(
    strategy: &Strategy,
    out_grad: InputBinding,
    weights: TensorBinding,
    in_grad: [usize; 4],
    args: ConvolutionArgs<2>,
) -> String {
    let client = client();
    let in_grad = TestInput::builder(client.clone(), in_grad.to_vec())
        .dtype(F16)
        .zeros()
        .generate_without_host_data()
        .binding();
    let inputs = ConvolutionInputs::BackwardData {
        out_grad,
        weights: InputBinding::new(weights, F16),
        in_grad,
    };
    match launch_ref::<2>(strategy, &client, inputs, args, dtypes()) {
        Ok(()) => panic!("the launch was accepted"),
        Err(e) => format!("{e:?}"),
    }
}

const SAME_ARGS: ConvolutionArgs<2> = ConvolutionArgs {
    stride: [1, 1],
    padding: [1, 1],
    dilation: [1, 1],
};

#[test]
fn a_strided_data_backward_is_refused_by_name() {
    let client = client();
    let out_grad = InputBinding::new(dense(&client, [1, 3, 3, 32]), F16);
    let args = ConvolutionArgs {
        stride: [2, 2],
        ..SAME_ARGS
    };
    let message = data_grad_refusal(
        &tuned(),
        out_grad,
        dense(&client, [32, 3, 3, 64]),
        [1, 6, 6, 64],
        args,
    );
    assert!(message.contains("stride of 1"), "{message}");
}

#[test]
fn a_quantized_data_backward_out_grad_is_refused_by_name() {
    let client = client();
    let data = dense(&client, [1, 6, 6, 32]);
    let out_grad = InputBinding::Quantized {
        scale: data.clone(),
        data,
        data_dtype: F16,
        scale_dtype: F32,
        shape: Shape::from(vec![1, 6, 6, 32]),
        scheme: Default::default(),
    };
    let weights = dense(&client, [32, 3, 3, 64]);
    let message = data_grad_refusal(&tuned(), out_grad, weights, [1, 6, 6, 64], SAME_ARGS);
    assert!(message.contains("quantized"), "{message}");
}

#[test]
fn a_data_gradient_of_other_channels_is_refused_by_name() {
    let client = client();
    let out_grad = InputBinding::new(dense(&client, [1, 6, 6, 32]), F16);
    let weights = dense(&client, [32, 3, 3, 64]);
    let message = data_grad_refusal(&tuned(), out_grad, weights, [1, 6, 6, 16], SAME_ARGS);
    assert!(message.contains("no groups"), "{message}");
}

#[test]
fn a_data_gradient_of_other_filters_is_refused_by_name() {
    let client = client();
    let out_grad = InputBinding::new(dense(&client, [1, 6, 6, 32]), F16);
    let weights = dense(&client, [16, 3, 3, 64]);
    let message = data_grad_refusal(&tuned(), out_grad, weights, [1, 6, 6, 64], SAME_ARGS);
    assert!(message.contains("output channels"), "{message}");
}

#[test]
fn a_data_gradient_of_another_batch_is_refused_by_name() {
    let client = client();
    let out_grad = InputBinding::new(dense(&client, [2, 6, 6, 32]), F16);
    let weights = dense(&client, [32, 3, 3, 64]);
    let message = data_grad_refusal(&tuned(), out_grad, weights, [1, 6, 6, 64], SAME_ARGS);
    assert!(message.contains("batch"), "{message}");
}

/// A step of taps and output channels is declined through cmma by name.
#[test]
fn cmma_declines_a_data_backward_step_across_taps() {
    let client = client();
    let out_grad = InputBinding::new(dense(&client, [2, 6, 6, 3]), F16);
    let weights = dense(&client, [3, 3, 3, 32]);
    let message = data_grad_refusal(
        &pinned(Staged::F2X2_P2X2),
        out_grad,
        weights,
        [2, 6, 6, 32],
        SAME_ARGS,
    );
    assert!(message.contains("taps and channels"), "{message}");
}
