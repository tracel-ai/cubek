//! The input's gradient through a multi-level algorithm.

use cubek_convolution::ConvAlgorithm;
use cubek_test_utils::ScopeUnderTest;

use crate::convolution::multi_level::Forced;
use crate::convolution::{Conv, Convolves, Declines, F16, Pass, Values};

impl Conv<1> {
    /// A 1D convolution of 16 channels to 16 under a kernel of three, summed exactly.
    pub(crate) fn sixteen_channels() -> Self {
        Conv::of(F16)
            .channels(16)
            .out_channels(16)
            .kernel([3])
            .values(Values::Ramp)
    }
}

/// Strided by two, so input positions the stride skips over receive no gradient from some taps:
/// the coordinates the stride does not divide are masked.
#[test]
fn backward_data_masks_the_coordinates_the_stride_does_not_divide() {
    let cyclic = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleSyncCyclic),
        Declines::AsItStates,
    );
    let strided = Conv::sixteen_channels().spatial([16]).stride([2]);
    cyclic.check(cyclic.at(&[Pass::BackwardData], &[strided]));
}

/// Padded at the end alone, which only the output gradient's larger extent states: eight input
/// positions under a kernel of three give eight outputs, not six.
#[test]
fn backward_data_supports_end_only_padding() {
    let cyclic = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleSyncCyclic),
        Declines::AsItStates,
    );
    let end_padded = Conv::sixteen_channels().spatial([8]).padding([(0, 2)]);
    cyclic.check(cyclic.at(&[Pass::BackwardData], &[end_padded]));
}
