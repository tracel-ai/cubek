use cubek_std::{TileSize, tile::mma::MmaIOConfig};

use crate::definition::SwizzleModes;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaMatmulConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub mma_io_config: MmaIOConfig,
}
