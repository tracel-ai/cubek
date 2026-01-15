use crate::components::tile::{SharedTileConfig, TileConfig};

use crate::components::stage::SwizzleMode;
use crate::definition::{StageIdent, TileSize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InterleavedMatmulConfig {
    pub shared: SharedTileConfig,
}

impl InterleavedMatmulConfig {
    pub fn from_shared_tile_config(config: SharedTileConfig) -> Self {
        Self { shared: config }
    }
    pub fn elements_per_unit_m(&self) -> usize {
        self.elements_in_tile_m() as usize
    }

    pub fn elements_per_unit_n(&self) -> usize {
        self.elements_in_tile_n() as usize
    }

    pub fn local_tile_size(&self) -> TileSize {
        TileSize {
            m: self.elements_in_tile_m(),
            n: self.elements_in_tile_n(),
            k: self.elements_in_tile_k(),
        }
    }

    pub fn elements_per_unit_k(&self) -> usize {
        let k = self.shared.elements_in_tile_k() as usize;
        let plane_dim = self.plane_dim() as usize;
        assert!(
            k.is_multiple_of(plane_dim),
            "k must be divisible by plane_dim. Got k={:?}, plane_dim={:?}",
            k,
            plane_dim
        );

        k / plane_dim
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
