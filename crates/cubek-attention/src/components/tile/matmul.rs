use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout, SwizzleModes,
    tile::{
        BounceConfig, CmmaMatmul, CmmaTile, Plane, ProductType, RegisterMatmul, Tile,
        allocate_bounce_tile, cmma_allocate_acc, cmma_allocate_lhs, cmma_allocate_rhs,
        register_allocate_acc, register_allocate_lhs, register_allocate_rhs,
    },
};
use cubek_std::{TileSize, as_cmma_layout};

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
pub fn allocate_lhs<L: Numeric>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<L, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_lhs::<L, Plane>(MatrixLayout::RowMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_lhs::<L, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

#[cube]
pub fn allocate_rhs<R: Numeric>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<R, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_rhs::<R, Plane>(MatrixLayout::RowMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_rhs::<R, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

#[cube]
pub fn allocate_rhs_transposed<R: Numeric>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<R, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_rhs::<R, Plane>(MatrixLayout::ColMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_rhs::<R, Plane>(MatrixLayout::ColMajor, c)
        }
    }
}

#[cube]
pub fn allocate_acc<A: Numeric>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<A, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            cmma_allocate_acc::<A, Plane>(MatrixLayout::RowMajor, c.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_acc::<A, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

/// Allocates an accumulator tile that takes part in row-wise softmax/output
/// scaling. For the cmma path this is a `Tile::Bounce` (cmma + smem + LocalTile);
/// for the register path it falls back to `Tile::Register`.
#[cube]
pub fn allocate_acc_bouncing<A: Float>(
    #[comptime] matmul: AttentionTileMatmul,
    #[comptime] bounce: BounceConfig,
) -> Tile<A, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            let matrix = unsafe {
                cubecl::cmma::Matrix::<A>::uninitialized(
                    cubecl::cmma::MatrixIdent::Accumulator,
                    c.tile_size.m as usize,
                    c.tile_size.n as usize,
                    c.tile_size.k as usize,
                    cubecl::cmma::MatrixLayout::Undefined,
                )
            };
            let cmma = CmmaTile::<A> {
                matrix,
                matrix_layout: MatrixLayout::RowMajor,
                tile_size: c.tile_size,
            };
            allocate_bounce_tile::<A, Plane>(cmma, bounce)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_acc::<A, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

/// Allocates an LHS tile that takes part in row-wise softmax/output scaling
/// (i.e. the value-matmul lhs that receives the post-softmax cast-down values).
/// For the cmma path this is a `Tile::Bounce`.
#[cube]
pub fn allocate_lhs_bouncing<L: Float>(
    #[comptime] matmul: AttentionTileMatmul,
    #[comptime] bounce: BounceConfig,
) -> Tile<L, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            let matrix = unsafe {
                cubecl::cmma::Matrix::<L>::uninitialized(
                    cubecl::cmma::MatrixIdent::A,
                    c.tile_size.m as usize,
                    c.tile_size.n as usize,
                    c.tile_size.k as usize,
                    as_cmma_layout(MatrixLayout::RowMajor),
                )
            };
            let cmma = CmmaTile::<L> {
                matrix,
                matrix_layout: MatrixLayout::RowMajor,
                tile_size: c.tile_size,
            };
            allocate_bounce_tile::<L, Plane>(cmma, bounce)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_lhs::<L, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}
