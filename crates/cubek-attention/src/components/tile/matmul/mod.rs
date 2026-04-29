use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout, SwizzleModes,
    tile::{
        CmmaMatmul, Plane, ProductType, RegisterMatmul, Tile, cmma_allocate_lhs, cmma_allocate_rhs,
        register_allocate_acc, register_allocate_lhs, register_allocate_rhs,
    },
};
use cubek_std::{StageIdent, TileSize};

/// Attention's tile-level matmul configuration. Each variant carries the per-kind
/// config from cubek-std.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum AttentionTileMatmul {
    Cmma(CmmaMatmul),
    Register(RegisterMatmul),
}

impl AttentionTileMatmul {
    pub fn new_register_unit(tile_size: TileSize) -> Self {
        AttentionTileMatmul::Register(RegisterMatmul {
            tile_size,
            plane_dim: 1,
            swizzle_modes: SwizzleModes::default(),
            product_type: ProductType::Inner,
        })
    }

    pub fn new_cmma(tile_size: TileSize, plane_dim: u32) -> Self {
        AttentionTileMatmul::Cmma(CmmaMatmul {
            tile_size,
            plane_dim,
            swizzle_modes: SwizzleModes::default(),
        })
    }

    pub fn tile_size(&self) -> TileSize {
        match self {
            AttentionTileMatmul::Cmma(c) => c.tile_size,
            AttentionTileMatmul::Register(c) => c.tile_size,
        }
    }
}

#[cube]
pub fn allocate_lhs<L: Numeric, VL: Size>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<L, VL, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_lhs::<L, VL, Plane>(MatrixLayout::RowMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_lhs::<L, VL, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

#[cube]
pub fn allocate_rhs<R: Numeric, VR: Size>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<R, VR, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_rhs::<R, VR, Plane>(MatrixLayout::RowMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_rhs::<R, VR, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

#[cube]
pub fn allocate_rhs_transposed<R: Numeric, VR: Size>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<R, VR, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_rhs::<R, VR, Plane>(MatrixLayout::ColMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_rhs::<R, VR, Plane>(MatrixLayout::ColMajor, c)
        }
    }
}

#[cube]
pub fn allocate_acc<A: Numeric, VA: Size>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<A, VA, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cubek_std::tile::cmma_allocate_acc::<A, VA, Plane>(MatrixLayout::RowMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_acc::<A, VA, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

#[cube]
pub fn load_lhs<E: Numeric, ES: Size, L: Numeric, VL: Size, R: Numeric, A: Numeric>(
    source: &Tile<E, ES, Plane, ReadOnly>,
    dest: &mut Tile<L, VL, Plane, ReadWrite>,
) {
    dest.copy_from::<E, ES, L, R, A, ReadOnly>(source, StageIdent::Lhs);
}

#[cube]
pub fn load_rhs<E: Float, ES: Size, L: Numeric, R: Numeric, VR: Size, A: Numeric>(
    source: &Tile<E, ES, Plane, ReadOnly>,
    dest: &mut Tile<R, VR, Plane, ReadWrite>,
) {
    dest.copy_from::<E, ES, L, R, A, ReadOnly>(source, StageIdent::Rhs);
}

#[cube]
pub fn execute<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>(
    lhs: &Tile<L, VL, Plane, ReadWrite>,
    rhs: &Tile<R, VR, Plane, ReadWrite>,
    acc: &mut Tile<A, VA, Plane, ReadWrite>,
) {
    acc.mma(lhs, rhs);
}
