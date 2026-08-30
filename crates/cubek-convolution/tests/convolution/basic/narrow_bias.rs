//! Regressions for accumulator loading in narrow-output convolutions.

use cubek_convolution::ConvAlgorithm;
use cubek_matmul::multi_level::components::tile::TileMatmulKind;
use cubek_matmul::multi_level::{
    PartitionSize, StageSize, TileSize,
    definition::TilingScheme,
    stage::{SwizzleMode, SwizzleModes},
};

use super::common::{default_partition_buffering, f16_dtypes};
use crate::convolution::launcher_strategy::{ConvolutionCase, ConvolutionSize, test_algo_case};

fn etive_residual_tiling() -> TilingScheme {
    TilingScheme::builder()
        .with_tile_size(TileSize { m: 16, n: 8, k: 16 })
        .with_partition_size(PartitionSize { m: 1, n: 8, k: 2 })
        .with_stage_size(StageSize { m: 4, n: 1, k: 1 })
        .build()
        .unwrap()
}

#[cfg(feature = "heavy")]
#[test]
fn simple_n64_mma_with_bias() {
    test_algo_case(
        ConvAlgorithm::SimpleSyncStrided,
        f16_dtypes(),
        TileMatmulKind::Mma,
        etive_residual_tiling(),
        SwizzleModes {
            lhs: SwizzleMode::B64,
            rhs: SwizzleMode::B64,
            ..Default::default()
        },
        default_partition_buffering(),
        ConvolutionCase {
            size: ConvolutionSize {
                h: 8,
                w: 8,
                c: 64,
                out_c: 64,
            },
            batches: 1,
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
        },
        true,
    );
}
