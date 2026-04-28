use cubek_std::stage::SwizzleMode;

use crate::{
    components::tile_matmul::{
        InterleavedMatmulConfig, MmaMatmulConfig, PlaneVecMatInnerProductConfig,
        RegisterMatmulConfig, SharedTileConfig, TileConfig,
    },
    definition::StageIdent,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum DispatchConfig {
    Cmma(SharedTileConfig),
    Mma(MmaMatmulConfig),
    Register(RegisterMatmulConfig),
    PlaneVec(PlaneVecMatInnerProductConfig),
    Interleaved(InterleavedMatmulConfig),
}

impl TileConfig for DispatchConfig {
    fn plane_dim(&self) -> u32 {
        match self {
            DispatchConfig::Cmma(config) => config.plane_dim(),
            DispatchConfig::Mma(config) => config.plane_dim(),
            DispatchConfig::Register(config) => config.plane_dim(),
            DispatchConfig::PlaneVec(config) => config.plane_dim(),
            DispatchConfig::Interleaved(config) => config.plane_dim(),
        }
    }

    fn elements_in_tile_m(&self) -> u32 {
        match self {
            DispatchConfig::Cmma(config) => config.elements_in_tile_m(),
            DispatchConfig::Mma(config) => config.elements_in_tile_m(),
            DispatchConfig::Register(config) => config.elements_in_tile_m(),
            DispatchConfig::PlaneVec(config) => config.elements_in_tile_m(),
            DispatchConfig::Interleaved(config) => config.elements_in_tile_m(),
        }
    }

    fn elements_in_tile_n(&self) -> u32 {
        match self {
            DispatchConfig::Cmma(config) => config.elements_in_tile_n(),
            DispatchConfig::Mma(config) => config.elements_in_tile_n(),
            DispatchConfig::Register(config) => config.elements_in_tile_n(),
            DispatchConfig::PlaneVec(config) => config.elements_in_tile_n(),
            DispatchConfig::Interleaved(config) => config.elements_in_tile_n(),
        }
    }

    fn elements_in_tile_k(&self) -> u32 {
        match self {
            DispatchConfig::Cmma(config) => config.elements_in_tile_k(),
            DispatchConfig::Mma(config) => config.elements_in_tile_k(),
            DispatchConfig::Register(config) => config.elements_in_tile_k(),
            DispatchConfig::PlaneVec(config) => config.elements_in_tile_k(),
            DispatchConfig::Interleaved(config) => config.elements_in_tile_k(),
        }
    }

    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        match self {
            DispatchConfig::Cmma(config) => config.swizzle_mode(ident),
            DispatchConfig::Mma(config) => config.swizzle_mode(ident),
            DispatchConfig::Register(config) => config.swizzle_mode(ident),
            DispatchConfig::PlaneVec(config) => config.swizzle_mode(ident),
            DispatchConfig::Interleaved(config) => config.swizzle_mode(ident),
        }
    }
}
