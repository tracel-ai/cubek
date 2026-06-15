use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::definition::ConvBlueprint;

/// Top-level user-facing strategy for `launch_ref`.
///
/// `Specific` selects an algorithm and tile-matmul kind, letting the routine
/// infer the rest. `Forced` bypasses inference and uses the supplied blueprint
/// directly (the algorithm tag must still be provided so the kernel-side
/// generic dispatch can pick the right reading-strategy implementation).
#[derive(Clone, Debug)]
pub enum Strategy {
    /// User picks the algorithm and tile-matmul kind. Tiling/swizzle/etc. are
    /// inferred from the problem.
    Inferred {
        algorithm: ConvAlgorithm,
        tile_kind: AcceleratedTileKind,
    },
    /// User supplies a pre-built blueprint. The algorithm tag tells the launcher
    /// which kernel generic to instantiate; the tiling/swizzle/etc. come from
    /// the blueprint. The tile-matmul kind comes from the blueprint as well.
    Forced {
        algorithm: ConvAlgorithm,
        blueprint: ConvBlueprint,
    },
}

/// The convolution-side algorithm enum. Subsumes the previous
/// `ReadingStrategy` axis and the Simple/Specialized split. A single value
/// here picks one concrete `Routine` impl (see `crate::routines`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConvAlgorithm {
    SimpleSyncCyclic,
    SimpleSyncStrided,
    SimpleSyncTilewise,
    SimpleAsyncCyclic,
    SimpleAsyncStrided,
    SimpleAsyncTma,
    SpecializedAsyncCyclic,
    SpecializedAsyncStrided,
    SpecializedTma,
}

impl Display for ConvAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ConvAlgorithm::SimpleSyncCyclic => "simple_sync_cyclic",
            ConvAlgorithm::SimpleSyncStrided => "simple_sync_strided",
            ConvAlgorithm::SimpleSyncTilewise => "simple_sync_tilewise",
            ConvAlgorithm::SimpleAsyncCyclic => "simple_async_cyclic",
            ConvAlgorithm::SimpleAsyncStrided => "simple_async_strided",
            ConvAlgorithm::SimpleAsyncTma => "simple_async_tma",
            ConvAlgorithm::SpecializedAsyncCyclic => "specialized_async_cyclic",
            ConvAlgorithm::SpecializedAsyncStrided => "specialized_async_strided",
            ConvAlgorithm::SpecializedTma => "specialized_tma",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// Which tile matmul to use for accelerated algorithms.
pub enum AcceleratedTileKind {
    #[default]
    Cmma,
    Mma,
}

impl Display for AcceleratedTileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceleratedTileKind::Cmma => f.write_str("cmma"),
            AcceleratedTileKind::Mma => f.write_str("mma"),
        }
    }
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Inferred {
                algorithm,
                tile_kind,
            } => write!(f, "{algorithm}_{tile_kind}"),
            Strategy::Forced {
                algorithm,
                blueprint: _,
            } => write!(f, "{algorithm}_forced"),
        }
    }
}
