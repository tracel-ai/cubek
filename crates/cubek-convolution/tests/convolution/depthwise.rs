//! The depthwise routine, forward, over every tiling the catalogue offers, and its weight gradient.
//!
//! The tile suite next door (cubecl-tile's `tests/tile/depthwise.rs`) checks that the *DSL* can
//! express a depthwise convolution, and builds its own space to do it. This checks the space
//! `launch_depthwise` actually builds, which is the one that ships, and the one whose tiling moves
//! whenever the kernel is optimized. A tiling that is fast and wrong has to fail here.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_convolution::components::ConvSetupError;
use cubek_convolution::{
    ConvolutionArgs, DepthwiseStrategy, DepthwiseTensors, DepthwiseTiling,
    DepthwiseWeightBackwardTensors, launch_depthwise, launch_depthwise_weight_backward,
};
use cubek_test_utils::{ScopeUnderTest, TestInput};

use crate::convolution::{
    Conv, ConvRoutine, ConvTensors, Convolves, Declines, F32, Pass, Storage, Values,
};

/// The depthwise forward routine at one tiling.
#[derive(Debug)]
struct Depthwise(DepthwiseStrategy);

impl ConvRoutine<2> for Depthwise {
    /// # Panics
    ///
    /// On a backward pass, which it does not compute.
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<2>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        assert_eq!(
            pass,
            Pass::Forward,
            "the depthwise routine computes the forward pass alone"
        );
        launch_depthwise(
            client,
            DepthwiseTensors {
                input: tensors.input.clone().binding(),
                weight: tensors.weight.clone().binding(),
                out: tensors.written.clone().binding(),
            },
            conv.args(),
            conv.group_count(),
            conv.element(),
            self.0,
        )
    }
}

/// The depthwise weight gradient, split over the positions.
#[derive(Debug)]
struct DepthwiseWeight;

impl ConvRoutine<2> for DepthwiseWeight {
    /// # Panics
    ///
    /// On a pass other than the weight's gradient, the only one it computes.
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<2>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        assert_eq!(
            pass,
            Pass::BackwardWeight,
            "it computes the weight's gradient alone"
        );
        launch_depthwise_weight_backward(
            client,
            DepthwiseWeightBackwardTensors {
                input: tensors.input.clone().binding(),
                out_grad: tensors.out_grad.clone().binding(),
                weight_grad: tensors.written.clone().binding(),
            },
            conv.args(),
            conv.group_count(),
            conv.element(),
        )
    }
}

impl Conv<2> {
    /// A depthwise convolution of `element`, summed exactly: give it as many groups as channels.
    fn depthwise(element: ElemType) -> Self {
        Conv::of(element).values(Values::Ramp)
    }

    /// This layer under every tiling the benchmark catalogue offers, plus the default. A tiling
    /// only the sweep reaches is still a tiling that can be selected, so it is still one that has
    /// to be right.
    fn check_every_tiling(self) {
        let mut tilings = vec![DepthwiseStrategy::Fixed(DepthwiseTiling::default())];
        for rows in [1, 2, 4] {
            for cols in [1, 3, 4] {
                for (chans, lines) in [(1, 1), (2, 1), (3, 1), (1, 2), (1, 4), (2, 4)] {
                    tilings.push(DepthwiseStrategy::Fixed(DepthwiseTiling {
                        rows,
                        cols,
                        chans,
                        lines,
                    }));
                }
            }
        }
        for tiling in tilings {
            let depthwise = Convolves::new(Depthwise(tiling), Declines::Never);
            depthwise.check(depthwise.at(&[Pass::Forward], &[self]));
        }
    }

    /// This layer under `tiling` alone.
    fn check_at(self, tiling: DepthwiseTiling) {
        let depthwise =
            Convolves::new(Depthwise(DepthwiseStrategy::Fixed(tiling)), Declines::Never);
        depthwise.check(depthwise.at(&[Pass::Forward], &[self]));
    }
}

/// The shape a depthwise layer actually has: a 3x3 window, unit stride, padded to keep the map.
#[test]
fn depthwise_3x3_padded() {
    Conv::depthwise(F32)
        .batches(2)
        .spatial([7, 7])
        .channels(40)
        .out_channels(40)
        .groups(40)
        .kernel([3, 3])
        .padding([(1, 1); 2])
        .check_every_tiling();
}

/// Unpadded, which takes the `checked` guard off every read.
#[test]
fn depthwise_3x3_unpadded() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([6, 6])
        .channels(33)
        .out_channels(33)
        .groups(33)
        .kernel([3, 3])
        .check_every_tiling();
}

/// Strided, which is how the encoder's early stages downsample.
#[test]
fn depthwise_3x3_stride_2() {
    Conv::depthwise(F32)
        .batches(2)
        .spatial([9, 9])
        .channels(32)
        .out_channels(32)
        .groups(32)
        .kernel([3, 3])
        .stride([2, 2])
        .padding([(1, 1); 2])
        .check_every_tiling();
}

/// A dilated 5x5: the shape seven of B4's blocks run, and the widest halo here.
#[test]
fn depthwise_5x5_dilated() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([8, 8])
        .channels(64)
        .out_channels(64)
        .groups(64)
        .kernel([5, 5])
        .dilation([2, 2])
        .padding([(4, 4); 2])
        .check_every_tiling();
}

/// A channel count that is not a multiple of the unit count, so the last cube's channel tile is
/// short and the walk has to stop where the tensor does.
#[test]
fn depthwise_channels_not_a_whole_number_of_cubes() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([5, 5])
        .channels(37)
        .out_channels(37)
        .groups(37)
        .kernel([3, 3])
        .padding([(1, 1); 2])
        .check_every_tiling();
}

/// End-only padding extends the output without changing the projection offset. The output size
/// divides every spatial tile used here, so correctness can't come from a ragged-tile guard.
#[test]
fn depthwise_3x3_end_padding() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([8, 8])
        .channels(32)
        .out_channels(32)
        .groups(32)
        .kernel([3, 3])
        .padding([(0, 2); 2])
        .check_every_tiling();
}

/// Re-laying the filter must follow the binding's actual address map, not assume contiguous
/// `[C, kh, kw, 1]` storage.
#[test]
fn depthwise_strided_weight() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([5, 5])
        .channels(8)
        .out_channels(8)
        .groups(8)
        .kernel([3, 3])
        .padding([(1, 1); 2])
        .stored(Storage::Gapped)
        .check_at(DepthwiseTiling::default());
}

/// Batch-one weight gradients use the output gradient as the filter. This 20x20 filter used to
/// expand all 400 taps into the kernel, making SPIR-V compilation stall in Mem2Reg.
#[test]
fn depthwise_large_filter_weight_gradient() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([20, 20])
        .channels(184)
        .out_channels(184)
        .groups(184)
        .kernel([20, 20])
        .padding([(1, 1); 2])
        .check_at(DepthwiseTiling {
            rows: 4,
            cols: 2,
            chans: 1,
            lines: 1,
        });
}

/// An unpadded large filter with vectorized channels.
#[test]
fn depthwise_large_filter_unpadded() {
    Conv::depthwise(F32)
        .batches(1)
        .spatial([23, 23])
        .channels(256)
        .out_channels(256)
        .groups(256)
        .kernel([20, 20])
        .check_at(DepthwiseTiling {
            rows: 4,
            cols: 4,
            chans: 1,
            lines: 4,
        });
}

/// The filter must carry exactly one channel's filter for every input/output channel. Reject the
/// mismatch before re-laying the binding under a larger logical shape.
#[test]
fn depthwise_rejects_mismatched_weight_shape() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let input_channels = 8;

    let input = TestInput::builder(client.clone(), Shape::new([1, 5, 5, input_channels]))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), Shape::new([1, 5, 5, input_channels]))
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    for (weight_shape, want_weight_channels, want_group_channels) in
        [([7, 3, 3, 1], 7, 1), ([8, 3, 3, 2], 8, 2)]
    {
        let weight = TestInput::builder(client.clone(), Shape::new(weight_shape))
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();
        let result = launch_depthwise(
            &client,
            DepthwiseTensors {
                input: input.clone().binding(),
                weight: weight.binding(),
                out: out.clone().binding(),
            },
            ConvolutionArgs::<2> {
                stride: [1; 2],
                padding: [1; 2],
                dilation: [1; 2],
            },
            input_channels,
            dtype,
            DepthwiseStrategy::Routine,
        );

        assert!(matches!(
            result,
            Err(ConvSetupError::NotDepthwise {
                groups: 8,
                input_channels: 8,
                output_channels: 8,
                weight_channels,
                weight_group_channels,
            }) if weight_channels == want_weight_channels
                && weight_group_channels == want_group_channels
        ));
    }
}

/// The weight gradient split over the positions, against the defining sum: channel multipliers,
/// strides, dilations, uneven paddings, more than one split with a partial last one, and partial
/// channel and spatial tiles; its tensors channels last, channels first, and gapped beside a
/// channels-first output gradient. Declined where a cube holds fewer than 128 units.
#[test]
fn depthwise_weight_split_matches_reference() {
    // `(shape NHWC, kernel, stride, dilation, (beginning, end) padding, multiplier)`.
    let layers = [
        ([1, 1, 1, 1], [1, 1], [1, 1], [1, 1], [(0, 0), (0, 0)], 1),
        ([2, 7, 9, 5], [3, 3], [1, 1], [1, 1], [(1, 1), (1, 1)], 1),
        ([3, 9, 11, 3], [2, 3], [2, 1], [2, 1], [(0, 1), (2, 0)], 2),
        ([2, 5, 7, 4], [3, 2], [2, 2], [2, 2], [(3, 0), (0, 2)], 1),
        // Exactly one full split, followed below by a partial second split.
        ([2, 32, 64, 3], [3, 3], [1, 1], [1, 1], [(1, 1), (1, 1)], 1),
        // More than one 4096-sample split, with a partial final split.
        ([4, 33, 35, 7], [3, 3], [1, 1], [1, 1], [(1, 1), (1, 1)], 1),
        // Channel tiles: partial channel and spatial tiles, then a multiplier with stride and
        // dilation.
        ([2, 17, 19, 35], [3, 3], [1, 1], [1, 1], [(1, 1), (1, 1)], 1),
        ([2, 19, 21, 33], [2, 3], [2, 1], [2, 2], [(1, 0), (0, 2)], 2),
    ];
    let convs: Vec<Conv<2>> = layers
        .into_iter()
        .flat_map(
            |(
                [batches, height, width, channels],
                kernel,
                stride,
                dilation,
                padding,
                multiplier,
            )| {
                let layer = Conv::depthwise(F32)
                    .batches(batches)
                    .spatial([height, width])
                    .channels(channels)
                    .out_channels(channels * multiplier)
                    .groups(channels)
                    .kernel(kernel)
                    .stride(stride)
                    .dilation(dilation)
                    .padding(padding);
                [
                    layer.stored(Storage::Dense),
                    layer.stored(Storage::ChannelsFirst),
                    layer
                        .stored(Storage::Gapped)
                        .out_grad_stored(Storage::ChannelsFirst),
                ]
            },
        )
        .collect();
    let weight = Convolves::new(DepthwiseWeight, Declines::AsItStates);
    weight.check(weight.at(&[Pass::BackwardWeight], &convs));
}

/// A dense convolution, and any element but `f32`, are declined.
#[test]
fn depthwise_weight_declines_dense_and_other_dtypes() {
    let client = &cubecl::test_device().client();
    for (groups, dtype) in [
        (1, f32::elem_type_native()),
        (4, half::f16::elem_type_native()),
    ] {
        let input = TensorHandle::empty(client, [1, 5, 5, 4], dtype);
        let out_grad = TensorHandle::empty(client, [1, 3, 3, 4], dtype);
        let weight_grad = TensorHandle::empty(client, [4, 3, 3, 1], dtype);
        assert!(
            launch_depthwise_weight_backward(
                client,
                DepthwiseWeightBackwardTensors {
                    input: input.binding(),
                    out_grad: out_grad.binding(),
                    weight_grad: weight_grad.binding(),
                },
                ConvolutionArgs {
                    stride: [1, 1],
                    padding: [0, 0],
                    dilation: [1, 1]
                },
                groups,
                dtype,
            )
            .is_err()
        );
    }
}
