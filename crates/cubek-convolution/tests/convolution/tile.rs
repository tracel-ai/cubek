//! The tiled routines against the host's convolution: each pass through each routine, over
//! windows, extents and storages that the boxes overhang or fill exactly.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::zspace::Strides;
use cubek_convolution::components::ConvSetupError;
use cubek_convolution::eval::benchmarks::settings;
use cubek_convolution::kernels::tiled::{
    ConvDecline, ConvRoutineKind, CubeStagedConvCandidate, CubeStagedConvSettings,
};
use cubek_convolution::{ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref};
use cubek_matmul::definition::{MatmulAvailabilityError, MatmulSetupError};
use cubek_std::routine::ROUTINE_LISTS_NO_SETTING;
use cubek_std::{Hardware, InputBinding};
use cubek_test_utils::{HostData, HostDataType, ScopeChecker};

use crate::convolution::{
    ByStrategy, Conv, ConvCase, ConvRoutine, ConvTensors, Convolves, Declines, F16, F32, Pass,
    Storage, Values,
};
use Pass::{BackwardData, BackwardWeight, Forward};

/// A value no result holds, which the output starts as so that a cell the kernel leaves alone
/// shows.
const UNTOUCHED: f32 = 100.0;

type Window = ConvolutionArgs<2>;

const fn window<const N: usize>(
    stride: [usize; N],
    padding: [usize; N],
    dilation: [usize; N],
) -> ConvolutionArgs<N> {
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
const SAME_1D: ConvolutionArgs<1> = window([1], [1], [1]);
const VALID_1D: ConvolutionArgs<1> = window([1], [0], [1]);
const SAME_3D: ConvolutionArgs<3> = window([1; 3], [1; 3], [1; 3]);
const STRIDE_2_3D: ConvolutionArgs<3> = window([2; 3], [1; 3], [1; 3]);
const ASYMMETRIC_3D: ConvolutionArgs<3> = window([1, 2, 1], [0, 1, 2], [2, 1, 1]);
const ASYMMETRIC_STRIDE_1_3D: ConvolutionArgs<3> = window([1; 3], [0, 1, 2], [2, 1, 1]);

/// A convolution of `element`, `channels` in and `filters` out, whose values the host sums
/// exactly.
fn conv_of<const N: usize>(
    element: ElemType,
    batch: usize,
    (channels, filters): (usize, usize),
    input: [usize; N],
    kernel: [usize; N],
    args: ConvolutionArgs<N>,
) -> Conv<N> {
    Conv::of(element)
        .batches(batch)
        .spatial(input)
        .channels(channels)
        .out_channels(filters)
        .kernel(kernel)
        .stride(args.stride)
        .dilation(args.dilation)
        .padded(args.padding)
        .values(Values::Ramp)
}

/// A convolution of half precision.
fn conv<const N: usize>(
    batch: usize,
    channels: (usize, usize),
    input: [usize; N],
    kernel: [usize; N],
    args: ConvolutionArgs<N>,
) -> Conv<N> {
    conv_of(F16, batch, channels, input, kernel, args)
}

/// [`conv`] over `f32`, which the CPU runtime runs.
fn conv32<const N: usize>(
    batch: usize,
    channels: (usize, usize),
    input: [usize; N],
    kernel: [usize; N],
    args: ConvolutionArgs<N>,
) -> Conv<N> {
    conv_of(F32, batch, channels, input, kernel, args)
}

fn pinned(settings: CubeStagedConvSettings) -> Strategy {
    Strategy::Tile(ConvRoutineKind::CubeStaged(Some(settings)))
}

/// The plane window routine, which reads every operand from global memory.
fn plane_window() -> Strategy {
    Strategy::Tile(ConvRoutineKind::PlaneWindow)
}

/// The first setting the device lists that serves.
fn first_served() -> Strategy {
    Strategy::Tile(ConvRoutineKind::CubeStaged(None))
}

/// The refusals a tile routine may answer a case with, by the reason it states.
#[derive(Debug, Clone, Copy)]
enum Accepts {
    /// A device that offers the setting no instruction.
    Instruction,
    /// Every refusal the routine states for a setting its device lists.
    Listed,
    /// A device that lists the routine no setting, or offers the setting no instruction.
    Unlisted,
}

impl Accepts {
    fn covers(self, error: &ConvSetupError) -> bool {
        let message = format!("{error:?}");
        let states = |refusal| message.contains(why(refusal));
        match self {
            Self::Instruction => {
                states(ConvDecline::InstructionUnoffered) || states(ConvDecline::RowsNotPicked)
            }
            Self::Listed => Self::Instruction.covers(error) || Self::listed_alone(error),
            Self::Unlisted => {
                Self::Instruction.covers(error) || message.contains(why(ROUTINE_LISTS_NO_SETTING))
            }
        }
    }

    /// The refusals only a listed setting meets: a holder other than a plane, and a stage past
    /// the device's shared memory.
    fn listed_alone(error: &ConvSetupError) -> bool {
        format!("{error:?}").contains(why(ConvDecline::WgmmaUnserved))
            || matches!(
                error,
                ConvSetupError::Matmul(MatmulSetupError::Unavailable(
                    MatmulAvailabilityError::SharedMemoryTooBig { .. }
                ))
            )
    }
}

/// A tile strategy a device may decline for the reasons `accepts` names; any other refusal fails
/// the test.
#[derive(Debug)]
struct Tile {
    strategy: Strategy,
    accepts: Accepts,
}

impl<const N: usize> ConvRoutine<N> for Tile {
    /// # Panics
    ///
    /// On a refusal `accepts` does not name.
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<N>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        let launched = launch_ref(
            &self.strategy,
            client,
            tensors.inputs(conv, pass),
            conv.args(),
            conv.matmul_elems(),
        );
        launched.inspect_err(|error| {
            assert!(
                self.accepts.covers(error),
                "a setting that serves this problem refused it: {error:?}"
            )
        })
    }
}

/// `conv`'s `pass` through `strategy` matches the host's, where the device serves it.
fn check<const N: usize>(pass: Pass, strategy: Strategy, accepts: Accepts, conv: Conv<N>) {
    let scope = Convolves::new(Tile { strategy, accepts }, Declines::AsItStates);
    scope.check([ConvCase { conv, pass }]);
}

/// A module of tests with one test for each row, `name: strategy, accepts, conv`: a launch that
/// faults poisons the device for its process, so no two rows share a test.
macro_rules! cases {
    ($table:ident, $pass:ident; $($name:ident: $strategy:expr, $accepts:ident, $conv:expr;)*) => {
        mod $table {
            use super::*;

            $(#[test]
            fn $name() {
                check($pass, $strategy, Accepts::$accepts, $conv);
            })*
        }
    };
}

/// A module with a test for each row, `name: conv`, over the 2x2 grid of 16 row tiles.
macro_rules! cube {
    ($table:ident, $pass:ident; $($name:ident: $conv:expr;)*) => {
        cases! {
            $table, $pass;
            $($name: pinned(settings::F2X2_P2X2_T16), Instruction, $conv;)*
        }
    };
}

/// The cube staged routine, through the instruction the tile picks for the device.
mod cube {
    use super::*;

    cube! {
        forward, Forward;
        whole_channels: conv(2, (64, 64), [9, 9], [3, 3], SAME);
        stride_2: conv(2, (64, 32), [12, 15], [3, 3], STRIDE_2);
        dilation_2: conv(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        stride_unequal: conv(2, (64, 32), [12, 15], [3, 3], window([1, 2], [1, 1], [1, 1]));
        dilation_unequal: conv(2, (64, 32), [12, 12], [3, 3], DILATION_UNEQUAL);
        no_padding_overhanging_width: conv(2, (64, 32), [5, 40], [3, 3], VALID);
        overhanging_filters: conv(2, (64, 10), [20, 20], [3, 3], SAME);
        overhanging_channels: conv(2, (12, 16), [9, 9], [3, 3], SAME);
        channels_first: conv(2, (64, 32), [6, 7], [3, 3], SAME).stored(Storage::ChannelsFirst);
        padded_channels: conv(2, (60, 32), [6, 7], [3, 3], SAME).stored(Storage::Padded);
        gapped: conv(2, (64, 32), [6, 7], [3, 3], SAME).stored(Storage::Gapped);
        offset: conv(2, (64, 32), [6, 7], [3, 3], SAME).stored(Storage::Offset);
        conv_1d_overhanging: conv(2, (64, 32), [37], [3], SAME_1D);
        conv_1d_three_batches: conv(3, (64, 32), [37], [3], SAME_1D);
        conv_3d_stride_2: conv(2, (32, 32), [5, 6, 7], [3, 3, 3], STRIDE_2_3D);
        conv_3d_asymmetric: conv(2, (32, 32), [7, 9, 7], [3, 3, 3], ASYMMETRIC_3D);
        conv_3d_channels_first: conv(2, (32, 32), [5, 6, 7], [3, 3, 3], SAME_3D).stored(Storage::ChannelsFirst);
    }

    cube! {
        weight_grad, BackwardWeight;
        whole_channels: conv(2, (64, 64), [4, 12], [3, 3], SAME);
        stride_2: conv(2, (64, 32), [6, 20], [3, 3], STRIDE_2);
        dilation_2: conv(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        dilation_unequal: conv(2, (64, 32), [12, 12], [3, 3], DILATION_UNEQUAL);
        no_padding: conv(2, (32, 32), [5, 12], [3, 3], VALID);
        overhang_channels_and_filters: conv(2, (40, 48), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_3: conv(2, (3, 10), [6, 12], [3, 3], SAME);
        batch_3: conv(3, (32, 16), [4, 12], [3, 3], SAME);
        narrow_map: conv(2, (32, 16), [6, 3], [3, 3], SAME);
        channels_first: conv(2, (64, 32), [5, 12], [3, 3], SAME).stored(Storage::ChannelsFirst);
        padded_out_grad: conv(2, (64, 30), [5, 12], [3, 3], SAME).out_grad_stored(Storage::Padded);
        gapped_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME).out_grad_stored(Storage::Gapped);
        offset: conv(2, (64, 32), [5, 12], [3, 3], SAME).stored(Storage::Offset);
        conv_1d_overhanging: conv(2, (64, 32), [37], [3], SAME_1D);
        conv_1d_short_output_beside_batch: conv(3, (32, 32), [7], [3], VALID_1D);
        conv_3d_stride_2: conv(2, (32, 32), [5, 6, 7], [3, 3, 3], STRIDE_2_3D);
        conv_3d_asymmetric: conv(2, (32, 32), [7, 9, 7], [3, 3, 3], ASYMMETRIC_3D);
    }

    cube! {
        data_grad, BackwardData;
        whole_channels: conv(2, (64, 64), [6, 12], [3, 3], SAME);
        no_padding: conv(2, (32, 32), [7, 12], [3, 3], VALID);
        dilation_2: conv(2, (64, 32), [8, 17], [3, 3], DILATION_2);
        overhanging_filters: conv(2, (32, 24), [6, 12], [3, 3], SAME);
        channels_3: conv(2, (3, 16), [6, 12], [3, 3], SAME);
        batch_3: conv(3, (32, 16), [4, 12], [3, 3], SAME);
        non_square_kernel: conv(2, (32, 24), [6, 12], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_first_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME).out_grad_stored(Storage::ChannelsFirst);
        padded_out_grad: conv(2, (64, 30), [5, 12], [3, 3], SAME).out_grad_stored(Storage::Padded);
        gapped_out_grad: conv(2, (64, 32), [5, 12], [3, 3], SAME).out_grad_stored(Storage::Gapped);
        offset: conv(2, (64, 32), [5, 12], [3, 3], SAME).stored(Storage::Offset);
        conv_1d_overhanging: conv(2, (64, 32), [37], [3], SAME_1D);
        conv_3d_padded: conv(2, (32, 32), [5, 6, 7], [3, 3, 3], SAME_3D);
        conv_3d_asymmetric: conv(2, (32, 32), [7, 9, 7], [3, 3, 3], ASYMMETRIC_STRIDE_1_3D);
    }

    // A few channels make a step of taps, which the fragment API declines by name and a device
    // whose tile picks the manual mma serves.
    cases! {
        taps, Forward;
        channels_1: pinned(settings::F2X2_P2X2_T16), Instruction, conv(1, (1, 16), [12, 12], [3, 3], SAME);
        large_kernel: pinned(settings::F2X2_P2X2_T16), Instruction, conv(1, (3, 16), [23, 23], [11, 11], window([4, 4], [1, 1], [1, 1]));
    }

    cases! {
        taps_data_grad, BackwardData;
        filters_3: pinned(settings::F2X2_P2X2_T16), Instruction, conv(2, (32, 3), [6, 12], [3, 3], SAME);
    }

    // Planes across the rows of a cube, over a map they overhang, in every direction.
    cases! {
        planes_across_forward, Forward;
        two: pinned(settings::F1X2_P1X2_T16.across(2)), Instruction, conv(2, (64, 64), [8, 7], [3, 3], SAME);
        four: pinned(settings::F1X2_P1X2_T16.across(4)), Instruction, conv(2, (64, 64), [8, 7], [3, 3], VALID);
    }

    cases! {
        planes_across_weight_grad, BackwardWeight;
        two: pinned(settings::F1X2_P1X2_T16.across(2)), Instruction, conv(2, (16, 64), [8, 9], [3, 3], SAME);
        four: pinned(settings::F1X2_P1X2_T16.across(4)), Instruction, conv(2, (16, 64), [8, 9], [3, 3], VALID);
    }

    cases! {
        planes_across_data_grad, BackwardData;
        two: pinned(settings::F1X2_P1X2_T16.across(2)), Instruction, conv(2, (32, 32), [6, 12], [3, 3], SAME);
        four: pinned(settings::F1X2_P1X2_T16.across(4)), Instruction, conv(2, (32, 32), [6, 12], [3, 3], SAME);
    }

    // Whole steps read at width one, and the 56 row setting over a map it overhangs.
    cases! {
        other_settings, Forward;
        width_one: pinned(settings::F2X2_P2X2_T16.width_one()), Instruction, conv(2, (64, 64), [9, 9], [3, 3], SAME);
        rows_56_overhang: pinned(settings::F7X1_P1X2_T8), Instruction, conv(1, (64, 32), [3, 60], [3, 3], SAME);
    }

    cases! {
        first_served_forward, Forward;
        narrow_map: first_served(), Unlisted, conv(2, (64, 64), [7, 7], [3, 3], SAME);
    }

    /// Every setting the device lists, pinned, on one small problem of `pass`: it matches the
    /// host's result or is declined by name, and some setting serves the pass.
    fn every_listed_setting(pass: Pass) {
        let client = client();
        let list = CubeStagedConvCandidate::list(&Hardware::from_client(&client));
        // A device without planes, as a CPU runtime, lists no staged setting at all.
        if list.is_empty() {
            return;
        }
        let conv = conv(2, (64, 32), [9, 9], [3, 3], SAME);
        let mut served = 0;
        for candidate in &list {
            let strategy = pinned(*candidate.settings());
            let scope = Convolves::new(
                Tile {
                    strategy,
                    accepts: Accepts::Listed,
                },
                Declines::AsItStates,
            );
            let attempt = scope.attempt(&client, &ConvCase { conv, pass });
            served += usize::from(attempt.map(|outcome| outcome.enforce()).is_ok());
        }
        assert!(served > 0, "no listed setting serves {pass:?}");
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

/// The plane window routine in `f32`, which runs on the CPU runtime.
mod plane_window {
    use super::*;

    /// A row of a table: the plane window routine, which refuses nothing here.
    macro_rules! window {
        ($table:ident, $pass:ident; $($name:ident: $conv:expr;)*) => {
            mod $table {
                use super::*;

                $(#[test]
                fn $name() {
                    let scope = Convolves::new(ByStrategy(plane_window()), Declines::Never);
                    scope.check([ConvCase { conv: $conv, pass: Pass::$pass }]);
                })*
            }
        };
    }

    window! {
        forward, Forward;
        same_3x3: conv32(2, (64, 32), [9, 9], [3, 3], SAME);
        stride_2: conv32(2, (64, 32), [12, 15], [3, 3], STRIDE_2);
        dilation_2: conv32(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        no_padding: conv32(2, (64, 32), [5, 40], [3, 3], VALID);
        overhanging_channels: conv32(2, (12, 16), [9, 9], [3, 3], SAME);
        overhanging_filters: conv32(2, (64, 10), [9, 9], [3, 3], SAME);
        channels_1: conv32(1, (1, 16), [12, 12], [3, 3], SAME);
        stride_unequal: conv32(2, (64, 32), [12, 15], [3, 3], window([1, 2], [1, 1], [1, 1]));
        dilation_unequal: conv32(2, (64, 32), [12, 12], [3, 3], DILATION_UNEQUAL);
        ragged_57x5: conv32(2, (64, 32), [57, 5], [3, 3], SAME);
        conv_1d_overhanging: conv32(2, (64, 32), [57], [3], SAME_1D);
        conv_1d_three_batches: conv32(3, (64, 32), [57], [3], SAME_1D);
        conv_3d_stride_2: conv32(2, (16, 12), [5, 6, 7], [3, 3, 3], STRIDE_2_3D);
        conv_3d_asymmetric: conv32(2, (16, 12), [7, 9, 7], [3, 3, 3], ASYMMETRIC_3D);
    }

    window! {
        weight_grad, BackwardWeight;
        same_3x3: conv32(2, (64, 32), [6, 12], [3, 3], SAME);
        stride_2: conv32(2, (32, 32), [7, 21], [3, 3], STRIDE_2);
        dilation_2: conv32(2, (64, 32), [12, 12], [3, 3], DILATION_2);
        dilation_unequal: conv32(2, (64, 32), [12, 12], [3, 3], DILATION_UNEQUAL);
        overhanging_channels_and_filters: conv32(2, (40, 12), [9, 11], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_3: conv32(2, (3, 10), [6, 12], [3, 3], SAME);
        ragged_57x5: conv32(2, (64, 32), [57, 5], [3, 3], SAME);
        channels_first_output: conv32(1, (2, 2), [5, 5], [3, 3], VALID).stored(Storage::ChannelsFirst);
        conv_1d_overhanging: conv32(2, (64, 32), [57], [3], SAME_1D);
        conv_3d_stride_2: conv32(2, (16, 12), [5, 6, 7], [3, 3, 3], STRIDE_2_3D);
        conv_3d_asymmetric: conv32(2, (16, 12), [7, 9, 7], [3, 3, 3], ASYMMETRIC_3D);
    }

    window! {
        data_grad, BackwardData;
        same_3x3: conv32(2, (64, 32), [6, 12], [3, 3], SAME);
        no_padding: conv32(2, (32, 32), [7, 12], [3, 3], VALID);
        dilation_2: conv32(2, (64, 32), [8, 17], [3, 3], DILATION_2);
        overhanging_channels_and_filters: conv32(2, (40, 12), [9, 11], [3, 3], SAME);
        non_square_kernel: conv32(2, (32, 24), [6, 12], [5, 3], window([1, 1], [2, 1], [1, 1]));
        channels_3: conv32(2, (3, 16), [6, 12], [3, 3], SAME);
        ragged_57x5: conv32(2, (64, 32), [57, 5], [3, 3], SAME);
        conv_1d_overhanging: conv32(2, (64, 32), [57], [3], SAME_1D);
        conv_3d_padded: conv32(2, (16, 12), [5, 6, 7], [3, 3, 3], SAME_3D);
        conv_3d_asymmetric: conv32(2, (16, 12), [7, 9, 7], [3, 3, 3], ASYMMETRIC_STRIDE_1_3D);
    }
}

fn client() -> Client {
    cubecl::test_device().client()
}

/// `shape` of `element`, all `value`, stored dense.
fn filled(client: &Client, shape: &[usize], value: f32) -> TensorHandle {
    let values = vec![value; shape.iter().product()];
    Conv::<2>::of(F16).store(client, shape, &values, Storage::Dense)
}

/// The inputs of `pass`: its lhs and rhs, and the output it writes.
fn inputs(
    pass: Pass,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
) -> ConvolutionInputs {
    match pass {
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
    }
}

fn launch(
    strategy: &Strategy,
    client: &Client,
    inputs: ConvolutionInputs,
    args: Window,
) -> Result<(), String> {
    let elems = Conv::<2>::of(F16).matmul_elems();
    launch_ref::<2>(strategy, client, inputs, args, elems).map_err(|e| format!("{e:?}"))
}

/// The reason a refusal the routines name states.
fn why(declined: impl Into<Decline>) -> &'static str {
    match declined.into() {
        Decline::Unserved { why } => why,
        other => panic!("a refusal with no stated reason: {other:?}"),
    }
}

/// An output with no element has nothing to compute: the launch returns and writes nowhere, as
/// the sentinel the output's buffer starts as shows.
#[test]
fn an_empty_output_launches_nothing() {
    let client = client();
    let sentinel = filled(&client, &[1, 1, 1, 1], UNTOUCHED);
    let empty = TensorHandle::new(
        sentinel.handle.clone(),
        vec![1, 0, 0, 32],
        Strides::new(&[1, 1, 1, 1]),
        F16,
    );
    let dense = |shape: [usize; 4]| InputBinding::new(filled(&client, &shape, 1.0).binding(), F16);
    let operands = inputs(
        Forward,
        dense([1, 2, 2, 64]),
        dense([32, 3, 3, 64]),
        empty.binding(),
    );
    assert_eq!(launch(&first_served(), &client, operands, VALID), Ok(()));
    let after = HostData::from_tensor_handle(&client, sentinel, HostDataType::F32);
    assert_eq!(after.get_f32(&[0, 0, 0, 0]), UNTOUCHED);
}
