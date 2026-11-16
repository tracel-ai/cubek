//! The direct routine, forward only, grouped or not, with a bias or not.
//!
//! Most layers here have more input channels per group than a vector has lanes, on purpose: only
//! those take the path that blocks output channels, and a grid of narrow layers would pass however
//! that path summed.

use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubek_convolution::components::ConvSetupError;
use cubek_convolution::{DirectTensors, launch_direct};
use cubek_test_utils::ScopeUnderTest;

use crate::convolution::{
    Conv, ConvRoutine, ConvTensors, Convolves, Declines, F16, F32, Pass, Values,
};

/// The direct routine, which serves every layer here.
#[derive(Debug)]
struct Direct;

impl<const N: usize> ConvRoutine<N> for Direct {
    /// # Panics
    ///
    /// On a backward pass, which it does not compute.
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<N>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        assert_eq!(
            pass,
            Pass::Forward,
            "the direct routine computes the forward pass alone"
        );
        launch_direct::<N>(
            client,
            DirectTensors {
                input: tensors.input.clone().binding(),
                weight: tensors.weight.clone().binding(),
                bias: tensors.bias.clone().map(|bias| bias.binding()),
                out: tensors.written.clone().binding(),
            },
            conv.args(),
            conv.group_count(),
            conv.element(),
        )
    }
}

/// `(channels, out_channels, groups)`. The output counts reach output vectors of 16, 4, 2 and 1,
/// so a channel block that does not divide its vector, or drops a remainder, has a case to fail.
const CHANNELS: [(usize, usize, usize); 11] = [
    (3, 8, 1),
    (16, 16, 1),
    (32, 32, 1),
    (64, 64, 1),
    (64, 12, 1),
    (64, 6, 1),
    (64, 5, 1),
    (128, 32, 2),
    (32, 64, 2),
    (192, 48, 3),
    (104, 26, 1),
];

impl Conv<2> {
    /// Every layer of the grid in `element`, with a bias and without: each channel count of
    /// [`CHANNELS`] under a `3 × 3` kernel padded, strided, dilated and unpadded, a pointwise one
    /// and a `1 × 3` one, over two batches of a `6 × 6` map, summed exactly.
    fn direct_grid(element: ElemType) -> Vec<Self> {
        let geometries = [
            Conv::of(element).kernel([3, 3]).padded([1, 1]),
            Conv::of(element)
                .kernel([3, 3])
                .stride([2, 2])
                .padded([1, 1]),
            Conv::of(element)
                .kernel([3, 3])
                .dilation([2, 2])
                .padded([2, 2]),
            Conv::of(element).kernel([3, 3]),
            Conv::of(element).kernel([1, 1]),
            Conv::of(element).kernel([1, 3]).padded([1, 1]),
        ];
        CHANNELS
            .into_iter()
            .flat_map(|(channels, out_channels, groups)| {
                geometries.into_iter().flat_map(move |geometry| {
                    let layer = geometry
                        .batches(2)
                        .spatial([6, 6])
                        .channels(channels)
                        .out_channels(out_channels)
                        .groups(groups)
                        .values(Values::Ramp);
                    [layer, layer.with_bias()]
                })
            })
            .collect()
    }
}

#[test]
fn direct_f32_matches_the_reference() {
    let direct = Convolves::new(Direct, Declines::Never);
    direct.check(direct.at(&[Pass::Forward], &Conv::direct_grid(F32)));
}

/// An `f16` vector has twice the lanes of an `f32` one at the same load width, so the same layer
/// compiles a different kernel and needs twice the input channels to take the blocked path.
#[test]
fn direct_f16_matches_the_reference() {
    let direct = Convolves::new(Direct, Declines::Never);
    direct.check(direct.at(&[Pass::Forward], &Conv::direct_grid(F16)));
}
