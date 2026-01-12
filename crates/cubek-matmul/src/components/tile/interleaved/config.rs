use crate::components::tile::{SharedTileConfig, TileConfig};

use crate::components::stage::SwizzleMode;
use crate::definition::StageIdent;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InterleavedMatmulConfig {
    pub shared: SharedTileConfig,
}

impl InterleavedMatmulConfig {
    pub fn from_shared_tile_config(config: SharedTileConfig) -> Self {
        Self { shared: config }
    }
}

impl TileConfig for InterleavedMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim()
    }

    fn elements_in_tile_m(&self) -> u32 {
        self.shared.elements_in_tile_m()
    }

    fn elements_in_tile_n(&self) -> u32 {
        self.shared.elements_in_tile_n()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.shared.elements_in_tile_k()
    }

    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        self.shared.swizzle_mode(ident)
    }
}
