//! The weight's gradient through a multi-level algorithm.

use cubecl::ir::OpaqueType;
use cubek_convolution::ConvAlgorithm;
use cubek_matmul::multi_level::{
    PartitionSize, StageSize, components::tile::TileMatmulKind, definition::TilingScheme,
};
use cubek_test_utils::{ScopeUnderTest, TestOutcome, ValidationResult};

use crate::convolution::multi_level::Forced;
use crate::convolution::{Conv, Convolves, Declines, F16, Pass, Values};

/// Padded at the end alone, which only the output gradient's larger extent states: the three taps
/// then overlap eight, seven and six input positions.
#[test]
fn backward_weight_supports_end_only_padding() {
    let cyclic = Convolves::new(
        Forced::smallest(ConvAlgorithm::SimpleSyncCyclic),
        Declines::AsItStates,
    );
    let end_padded = Conv::sixteen_channels().spatial([8]).padding([(0, 2)]);
    cyclic.check(cyclic.at(&[Pass::BackwardWeight], &[end_padded]));
}

impl Conv<2> {
    /// A `1 × 1` convolution on a `1 × 1` map, 16 channels to 256, at ten batches: its weight
    /// gradient contracts over the batches alone, a matmul `256 × 16`, ten deep.
    fn pointwise_over_ten_batches() -> Self {
        Conv::of(F16)
            .batches(10)
            .channels(16)
            .out_channels(256)
            .values(Values::Ramp)
    }
}

impl Forced {
    /// This routine on the pointwise weight gradient, which it must run.
    fn runs_the_pointwise_weight_gradient(self) {
        let forced = Convolves::new(self, Declines::Never);
        forced.check(forced.at(
            &[Pass::BackwardWeight],
            &[Conv::pointwise_over_ten_batches()],
        ));
    }

    /// A strided reader at one `16 × 8 × 16` mma tile a stage, an rhs stage of sixteen vectors of
    /// eight halves for a plane of 32 units: a stage its loading units do not divide evenly, which
    /// it must load. Skipped where the device has no such mma, or `algorithm` no copy engine to
    /// fill with.
    fn strided_reader_on_an_uneven_stage(algorithm: ConvAlgorithm) {
        let client = cubecl::test_device().client();
        let elems = Conv::pointwise_over_ten_batches().matmul_elems();
        let features = &client.properties().features;
        let offered = features.matmul.mma.iter().any(|cfg| {
            cfg.a_type == elems.lhs_register
                && cfg.b_type == elems.rhs_register
                && cfg.cd_type == elems.acc_register
                && (cfg.m, cfg.n, cfg.k) == (16, 8, 16)
        });
        let async_copy = algorithm == ConvAlgorithm::SimpleAsyncStrided;
        let skip = |reason: &str| {
            TestOutcome::Validated(ValidationResult::Skipped(reason.to_string())).enforce();
        };
        if !offered {
            return skip("the device offers no 16x8x16 f16 mma");
        }
        if async_copy && !(features.copy_async && features.supports_type(OpaqueType::Barrier)) {
            return skip("the device has no async copy");
        }
        let plane_dim = client.properties().hardware.plane_size_max;
        let stage_vectors = 16 * 8 * elems.rhs_stage.size() as u32 / 16;
        assert!(
            !stage_vectors.is_multiple_of(plane_dim),
            "{stage_vectors} vectors split evenly over {plane_dim} units: the test would not \
             exercise an uneven split"
        );
        Forced {
            algorithm,
            tile_kind: TileMatmulKind::Mma,
            tiling: TilingScheme::builder()
                .with_tile_size((16, 8, 16).into())
                .with_partition_size(PartitionSize { m: 1, n: 1, k: 1 })
                .with_stage_size(StageSize { m: 1, n: 1, k: 1 })
                .build()
                .unwrap(),
        }
        .runs_the_pointwise_weight_gradient();
    }
}

#[test]
fn simple_sync_strided_loads_an_unevenly_split_stage() {
    Forced::strided_reader_on_an_uneven_stage(ConvAlgorithm::SimpleSyncStrided);
}

#[test]
fn simple_async_strided_loads_an_unevenly_split_stage() {
    Forced::strided_reader_on_an_uneven_stage(ConvAlgorithm::SimpleAsyncStrided);
}

/// A stage wider than the weight gradient's `n` writes only the gradient's own channels.
/// Regression for the transposed layout checking `n` with the `m` bounds flag, which zeroed the
/// gradient. Skipped where the device offers no `f16` cmma of the instruction tile.
#[test]
fn weight_gradient_stage_wider_than_its_channels_writes_only_its_channels() {
    let client = cubecl::test_device().client();
    let elems = Conv::pointwise_over_ten_batches().matmul_elems();
    let tile = Forced::instruction_tile();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == elems.lhs_register
            && cfg.b_type == elems.rhs_register
            && cfg.cd_type == elems.acc_register
            && (cfg.m, cfg.n, cfg.k) == (tile.m, tile.n, tile.k)
    });
    if !offered {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the device offers no f16 cmma of the instruction tile".to_string(),
        ))
        .enforce();
        return;
    }
    // Twice the gradient's 16 channels a stage.
    let stage_n = 2 * 16;
    Forced {
        algorithm: ConvAlgorithm::SimpleSyncCyclic,
        tile_kind: TileMatmulKind::Cmma,
        tiling: TilingScheme::builder()
            .with_tile_size(tile)
            .with_partition_size(PartitionSize {
                m: 1,
                n: (stage_n / tile.n) as u8,
                k: 1,
            })
            .with_stage_size(StageSize { m: 1, n: 1, k: 1 })
            .build()
            .unwrap(),
    }
    .runs_the_pointwise_weight_gradient();
}
