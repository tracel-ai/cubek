use cubek_std::TileSize;

use crate::definition::SwizzleModes;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CmmaMatmulConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl CmmaMatmulConfig {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_modes: SwizzleModes) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
        }
    }
}
