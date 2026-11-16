//! The unit routine, `Strategy::Unit`: an `f32` register-tiled implicit convolution for devices
//! without a usable matrix instruction. Every pass, of every rank, in the two storages a framework
//! hands it (channels last with padded rows, and channels first), the forward pass with a bias and
//! without, at the tilings the selector picks.

#[cfg(feature = "heavy")]
use cubek_convolution::UnitTiling;
use cubek_convolution::{Strategy, UnitStrategy};
use cubek_matmul::multi_level::routines::TileSizeSelection;
use cubek_test_utils::ScopeUnderTest;

use crate::convolution::{ByStrategy, Conv, ConvCase, Convolves, Declines, F32, Pass, Storage};

/// The selector's own tilings, run on every case.
const INFERRED: [UnitStrategy; 2] = [
    UnitStrategy::Inferred(TileSizeSelection::MinTileSize),
    UnitStrategy::Inferred(TileSizeSelection::MaxTileSize),
];

/// Every tiling, presets included. A preset only changes the matmul partition, and each one is a
/// set of kernels the CPU runtime compiles slowly, so the presets run on the small cases only.
#[cfg(feature = "heavy")]
const ALL: [UnitStrategy; 7] = [
    UnitStrategy::Inferred(TileSizeSelection::MinTileSize),
    UnitStrategy::Inferred(TileSizeSelection::MaxTileSize),
    UnitStrategy::Fixed(UnitTiling::NARROW),
    UnitStrategy::Fixed(UnitTiling::BALANCED),
    UnitStrategy::Fixed(UnitTiling::WIDE),
    UnitStrategy::Fixed(UnitTiling::DEEP),
    UnitStrategy::Fixed(UnitTiling::TALL),
];

impl<const N: usize> Conv<N> {
    /// Every pass of this convolution at each of `tilings`, in both storages, the forward pass
    /// with a bias and without, against the host.
    fn check_every_pass_either_way(self, tilings: &[UnitStrategy]) {
        let cases: Vec<ConvCase<N>> = [Storage::Padded, Storage::ChannelsFirst]
            .into_iter()
            .flat_map(|storage| {
                let conv = self.stored(storage);
                [
                    ConvCase {
                        conv,
                        pass: Pass::Forward,
                    },
                    ConvCase {
                        conv: conv.with_bias(),
                        pass: Pass::Forward,
                    },
                    ConvCase {
                        conv,
                        pass: Pass::BackwardData,
                    },
                    ConvCase {
                        conv,
                        pass: Pass::BackwardWeight,
                    },
                ]
            })
            .collect();
        for &tiling in tilings {
            let unit = Convolves::new(ByStrategy(Strategy::Unit(tiling)), Declines::Never);
            unit.check(cases.clone());
        }
    }
}

/// The light tier's one case: a padded `3 × 3` kernel over a `7 × 9` map, 4 channels to 8.
#[test]
fn unit_convolution_small_matches_reference() {
    Conv::of(F32)
        .batches(2)
        .spatial([7, 9])
        .channels(4)
        .out_channels(8)
        .kernel([3, 3])
        .padded([1, 1])
        .check_every_pass_either_way(&INFERRED);
}

#[cfg(feature = "heavy")]
#[test]
fn unit_convolution_matches_reference() {
    // A 1D convolution padded unevenly.
    Conv::of(F32)
        .batches(2)
        .spatial([13])
        .channels(3)
        .out_channels(7)
        .kernel([3])
        .stride([2])
        .dilation([2])
        .padding([(1, 2)])
        .check_every_pass_either_way(&INFERRED);
    Conv::of(F32)
        .batches(2)
        .spatial([7, 9])
        .channels(4)
        .out_channels(8)
        .kernel([3, 3])
        .stride([1, 1])
        .dilation([1, 1])
        .padding([(1, 1), (1, 1)])
        .check_every_pass_either_way(&INFERRED);
    Conv::of(F32)
        .batches(2)
        .spatial([9, 11])
        .channels(17)
        .out_channels(33)
        .kernel([2, 3])
        .stride([2, 1])
        .dilation([2, 2])
        .padding([(0, 1), (3, 0)])
        .check_every_pass_either_way(&INFERRED);
    Conv::of(F32)
        .batches(2)
        .spatial([5, 7])
        .channels(32)
        .out_channels(32)
        .kernel([3, 2])
        .stride([2, 2])
        .dilation([1, 2])
        .padding([(1, 0), (0, 2)])
        .check_every_pass_either_way(&INFERRED);
    Conv::of(F32)
        .batches(1)
        .spatial([4, 5, 7])
        .channels(4)
        .out_channels(5)
        .kernel([2, 3, 2])
        .stride([1, 2, 1])
        .dilation([2, 1, 2])
        .padding([(1, 0), (0, 1), (2, 0)])
        .check_every_pass_either_way(&INFERRED);
    // Past the split-K threshold, with an uneven final interval and tail channels.
    Conv::of(F32)
        .batches(2)
        .spatial([37, 61])
        .channels(3)
        .out_channels(7)
        .kernel([3, 3])
        .stride([1, 1])
        .dilation([1, 1])
        .padding([(1, 1), (1, 1)])
        .check_every_pass_either_way(&INFERRED);
    Conv::of(F32)
        .batches(1)
        .spatial([5003])
        .channels(4)
        .out_channels(8)
        .kernel([3])
        .stride([1])
        .dilation([1])
        .padding([(1, 1)])
        .check_every_pass_either_way(&INFERRED);
    Conv::of(F32)
        .batches(1)
        .spatial([2, 17, 137])
        .channels(3)
        .out_channels(7)
        .kernel([1, 3, 2])
        .stride([1, 1, 1])
        .dilation([1, 1, 1])
        .padding([(0, 0), (1, 1), (1, 0)])
        .check_every_pass_either_way(&INFERRED);
    // Many partials and a short final split, as large feature maps have.
    Conv::of(F32)
        .batches(4)
        .spatial([161, 163])
        .channels(4)
        .out_channels(4)
        .kernel([3, 3])
        .stride([1, 1])
        .dilation([1, 1])
        .padding([(1, 1), (1, 1)])
        .check_every_pass_either_way(&INFERRED);
}

/// The input's gradient pads the output channels, even where the input channels already equal
/// the padded output channels: comparing the input channels to the padding misses this tail.
#[cfg(feature = "heavy")]
#[test]
fn unit_convolution_pads_the_output_channels_at_every_tiling() {
    Conv::of(F32)
        .batches(2)
        .spatial([5, 7])
        .channels(8)
        .out_channels(7)
        .kernel([3, 3])
        .padded([1, 1])
        .check_every_pass_either_way(&ALL);
    Conv::of(F32)
        .batches(1)
        .spatial([9])
        .channels(4)
        .out_channels(3)
        .kernel([3])
        .stride([2])
        .padded([1])
        .check_every_pass_either_way(&ALL);
}
