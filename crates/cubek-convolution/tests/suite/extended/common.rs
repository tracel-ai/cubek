//! Shared helpers for the extended (forced-blueprint) tier.

use cubecl::prelude::CubePrimitive;
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems, TilingScheme};
use cubek_std::{PartitionSize, StageSize, TileSize};

use crate::suite::launcher_strategy::ConvolutionSize;

/// `MatmulElems` with f16 globals (and the f32 accumulator picked by
/// `from_globals`).
pub(crate) fn f16_dtypes() -> MatmulElems {
    let f16 = half::f16::as_type_native_unchecked().storage_type();
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f16,
        rhs: f16,
        out: f16,
    })
}

/// Tile size that works on both macOS (CMMA is 8x8x8) and elsewhere.
#[cfg(target_os = "macos")]
pub(crate) fn default_tile_size() -> TileSize {
    TileSize { m: 8, n: 8, k: 8 }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn default_tile_size() -> TileSize {
    TileSize {
        m: 16,
        n: 16,
        k: 16,
    }
}

pub(crate) fn tiling_scheme(
    tile: TileSize,
    partition: PartitionSize,
    stage: StageSize,
) -> TilingScheme {
    TilingScheme::builder()
        .with_tile_size(tile)
        .with_partition_size(partition)
        .with_stage_size(stage)
        .build()
        .unwrap()
}

/// Default 32x32x32x16 conv used for advanced-knob tests.
pub(crate) fn default_size() -> ConvolutionSize {
    ConvolutionSize {
        h: 32,
        w: 32,
        c: 32,
        out_c: 16,
    }
}
