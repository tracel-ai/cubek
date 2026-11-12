//! The forward pass of each multi-level algorithm, at its smallest tiling: a 2D convolution with a
//! dilated, unevenly padded kernel, and the same padded at the end alone, which a routine is told
//! only through the larger output it is handed.

use cubek_convolution::ConvAlgorithm;
use cubek_matmul::multi_level::PartitionSize;
use cubek_test_utils::ScopeUnderTest;

use crate::convolution::multi_level::Forced;
use crate::convolution::{Conv, Convolves, Declines, F16, Pass};

impl Conv<2> {
    /// Two batches of a `side × side` map under a `4 × 3` kernel dilated `3 × 2`, in `f16`.
    fn dilated_layer(side: usize) -> Self {
        Conv::of(F16)
            .batches(2)
            .spatial([side, side])
            .kernel([4, 3])
            .dilation([3, 2])
    }

    /// A `16 × 16` map of 16 channels to 32, padded `3 × 1` at both ends and at the end alone.
    fn small_layers() -> [Self; 2] {
        let small = Self::dilated_layer(16).channels(16).out_channels(32);
        [small.padded([3, 1]), small.padding([(0, 3), (0, 1)])]
    }
}

#[test]
fn simple_sync_cyclic_matches_the_host() {
    let cyclic = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleSyncCyclic),
        Declines::AsItStates,
    );
    cyclic.check(cyclic.at(&[Pass::Forward], &Conv::small_layers()));
}

#[test]
fn simple_sync_strided_matches_the_host() {
    let strided = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleSyncStrided),
        Declines::AsItStates,
    );
    strided.check(strided.at(&[Pass::Forward], &Conv::small_layers()[..1]));
}

/// Tilewise, its partitions two tiles along `m` and `n`.
#[test]
fn simple_sync_tilewise_matches_the_host() {
    let tilewise = Convolves::new(
        Forced::at(
            ConvAlgorithm::SimpleSyncTilewise,
            PartitionSize { m: 2, n: 2, k: 1 },
        ),
        Declines::AsItStates,
    );
    tilewise.check(tilewise.at(&[Pass::Forward], &Conv::small_layers()[..1]));
}

#[test]
fn simple_async_cyclic_matches_the_host() {
    let cyclic = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleAsyncCyclic),
        Declines::AsItStates,
    );
    cyclic.check(cyclic.at(&[Pass::Forward], &Conv::small_layers()));
}

#[test]
fn simple_async_strided_matches_the_host() {
    let strided = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleAsyncStrided),
        Declines::AsItStates,
    );
    strided.check(strided.at(&[Pass::Forward], &Conv::small_layers()[..1]));
}

#[test]
fn simple_async_tma_matches_the_host() {
    let tma = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleAsyncTma),
        Declines::AsItStates,
    );
    tma.check(tma.at(&[Pass::Forward], &Conv::small_layers()));
}

/// A wider layer, `32 × 32` of 32 channels to 16.
#[cfg(feature = "heavy")]
#[test]
fn the_sync_algorithms_match_the_host_on_a_wider_layer() {
    for algorithm in [
        ConvAlgorithm::SimpleSyncCyclic,
        ConvAlgorithm::SimpleSyncStrided,
    ] {
        let forced = Convolves::new(Forced::smallest(algorithm), Declines::AsItStates);
        let wide = Conv::dilated_layer(32).channels(32).out_channels(16);
        forced.check(forced.at(&[Pass::Forward], &[wide.padded([3, 1])]));
    }
}
