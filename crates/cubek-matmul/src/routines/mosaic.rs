use std::fmt::Display;

/// TODO kernel is temporary, so not much effort on blueprint
pub const DEFAULT_TILE_SIZE: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MosaicStrategy {
    /// Square sub-tile edge staged into shared memory. Must divide `M`, `N`,
    /// and `K`.
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
