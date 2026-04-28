use cubek_std::TileSize;

use crate::definition::SwizzleModes;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InterleavedMatmulConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl InterleavedMatmulConfig {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_modes: SwizzleModes) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
        }
    }

    pub fn elements_per_unit_m(&self) -> usize {
        self.tile_size.m() as usize
    }

    pub fn elements_per_unit_n(&self) -> usize {
        self.tile_size.n() as usize
    }

    pub fn local_tile_size(&self) -> TileSize {
        TileSize {
            m: self.tile_size.m(),
            n: self.tile_size.n(),
            k: self.tile_size.k(),
        }
    }

    pub fn elements_per_unit_k(&self) -> usize {
        let k = self.tile_size.k() as usize;
        let plane_dim = self.plane_dim as usize;
        assert!(
            k.is_multiple_of(plane_dim),
            "k must be divisible by plane_dim. Got k={:?}, plane_dim={:?}",
            k,
            plane_dim
        );

        k / plane_dim
    }
}
