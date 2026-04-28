//! Shared helpers for the basic tier.

use cubecl::prelude::CubePrimitive;
use cubek_matmul::{
    components::stage::PartitionBuffering,
    definition::{MatmulElems, MatmulGlobalElems, TilingScheme},
};
use cubek_std::{PartitionSize, StageSize, SwizzleModes, TileSize};

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

/// Tile size that works on both macOS (where CMMA is 8x8x8) and elsewhere.
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

/// Smallest viable partition for a smoke test.
pub(crate) fn small_partition() -> PartitionSize {
    PartitionSize { m: 1, n: 1, k: 1 }
}

/// Two-plane stage along m,n. Default for most basic-tier tests.
pub(crate) fn small_stage() -> StageSize {
    StageSize { m: 2, n: 2, k: 1 }
}

pub(crate) fn default_tiling_scheme() -> TilingScheme {
    TilingScheme::builder()
        .with_tile_size(default_tile_size())
        .with_partition_size(small_partition())
        .with_stage_size(small_stage())
        .build()
        .unwrap()
}

pub(crate) fn default_swizzle() -> SwizzleModes {
    SwizzleModes::default()
}

pub(crate) fn default_partition_buffering() -> PartitionBuffering {
    PartitionBuffering::Single
}

/// Smoke-sized 2D problem (16x16 spatial, 16 in_c, 32 out_c).
pub(crate) fn small_size() -> ConvolutionSize {
    ConvolutionSize {
        h: 16,
        w: 16,
        c: 16,
        out_c: 32,
    }
}

/// Wider mid-sized problem, used for `basic`-feature-gated tests.
pub(crate) fn medium_size() -> ConvolutionSize {
    ConvolutionSize {
        h: 32,
        w: 32,
        c: 32,
        out_c: 16,
    }
}
