//! Strategy knobs for the [`Mosaic`](crate::components::batch::mosaic) family.
//!
//! Mosaic launches tile-DSL tiles directly instead of going through the
//! `Routine`/`MatmulArgs` machinery, so it needs only its user-facing knobs
//! here — there is no `Blueprint`/`Config`/`Routine` to expand.

use std::fmt::Display;

/// The default square sub-tile edge (elements per axis) staged into shared
/// memory. Picked small enough to divide common test shapes; the launch errors
/// out when it doesn't divide `M`, `N`, and `K`.
pub const DEFAULT_TILE_SIZE: usize = 4;

/// User-facing knobs for the Mosaic matmul.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MosaicStrategy {
    /// Square sub-tile edge staged into shared memory. Must divide `M`, `N`,
    /// and `K`. A larger tile loads more work per stage (better reuse) at the
    /// cost of a larger shared-memory footprint.
    pub tile_size: usize,
}

impl Default for MosaicStrategy {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
        }
    }
}

impl Display for MosaicStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_t{}", self.tile_size)
    }
}
