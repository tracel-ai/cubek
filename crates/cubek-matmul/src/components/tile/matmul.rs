use crate::definition::StageIdent;
use cubek_std::{
    stage::SwizzleMode,
    tile::{
        cmma::CmmaMatmul, interleaved::InterleavedMatmul, mma::MmaMatmul,
        plane_vec_mat_inner_product::PlaneVecMatInnerProduct, register::RegisterMatmul,
    },
};

/// Tile-level matmul configuration. Each variant carries the per-kind config.
///
/// This is both the runtime selector and the comptime configuration: pick the
/// variant that matches the kernel you want, then forward the value into the
/// stage layer where its accessors drive allocation and execution.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileMatmul {
    Cmma(CmmaMatmul),
    Mma(MmaMatmul),
    Register(RegisterMatmul),
    PlaneVec(PlaneVecMatInnerProduct),
    Interleaved(InterleavedMatmul),
}

impl TileMatmul {
    pub fn plane_dim(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.plane_dim,
            TileMatmul::Mma(c) => c.plane_dim,
            TileMatmul::Register(c) => c.plane_dim,
            TileMatmul::PlaneVec(c) => c.plane_dim,
            TileMatmul::Interleaved(c) => c.plane_dim,
        }
    }

    pub fn elements_in_tile_m(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.tile_size.m(),
            TileMatmul::Mma(c) => c.tile_size.m(),
            TileMatmul::Register(c) => c.tile_size.m(),
            TileMatmul::PlaneVec(c) => c.tile_size.m(),
            TileMatmul::Interleaved(c) => c.tile_size.m(),
        }
    }

    pub fn elements_in_tile_n(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.tile_size.n(),
            TileMatmul::Mma(c) => c.tile_size.n(),
            TileMatmul::Register(c) => c.tile_size.n(),
            TileMatmul::PlaneVec(c) => c.tile_size.n(),
            TileMatmul::Interleaved(c) => c.tile_size.n(),
        }
    }

    pub fn elements_in_tile_k(&self) -> u32 {
        match self {
            TileMatmul::Cmma(c) => c.tile_size.k(),
            TileMatmul::Mma(c) => c.tile_size.k(),
            TileMatmul::Register(c) => c.tile_size.k(),
            TileMatmul::PlaneVec(c) => c.tile_size.k(),
            TileMatmul::Interleaved(c) => c.tile_size.k(),
        }
    }

    pub fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        let modes = match self {
            TileMatmul::Cmma(c) => c.swizzle_modes,
            TileMatmul::Mma(c) => c.swizzle_modes,
            TileMatmul::Register(c) => c.swizzle_modes,
            TileMatmul::PlaneVec(c) => c.swizzle_modes,
            TileMatmul::Interleaved(c) => c.swizzle_modes,
        };

        match ident {
            StageIdent::Lhs => modes.lhs,
            StageIdent::Rhs => modes.rhs,
            StageIdent::Acc => modes.acc,
            StageIdent::Out => modes.out,
        }
    }
}
