use cubek_std::stage::SwizzleMode;

use crate::{
    components::tile_matmul::{SharedTileConfig, TileConfig},
    definition::StageIdent,
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PlaneVecMatInnerProductConfig {
    pub shared: SharedTileConfig,
    pub reduce_vector_size: u32,
}

impl PlaneVecMatInnerProductConfig {
    pub fn new(shared: SharedTileConfig, reduce_vector_size: u32) -> Self {
        Self {
            shared,
            reduce_vector_size,
        }
    }
}

impl TileConfig for PlaneVecMatInnerProductConfig {
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

    fn reduce_vector_size(&self) -> u32 {
        self.reduce_vector_size
    }
}
