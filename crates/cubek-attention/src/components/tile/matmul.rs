use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout, SwizzleModes,
    tile::{
        BounceConfig, CmmaMatmul, CmmaTile, InnerLayout, Plane, ProductType, RegisterMatmul, Tile,
        allocate_bounce_tile, cmma_allocate_lhs, cmma_allocate_rhs, register_allocate_acc,
        register_allocate_lhs, register_allocate_rhs,
    },
};
use cubek_std::{TileSize, as_cmma_layout};

/// Cmma variant of [`AttentionTileMatmul`]. Carries the underlying [`CmmaMatmul`]
/// alongside the comptime parameters needed to build per-tile [`BounceConfig`]s
/// for the smem round-trip used by row-wise softmax/output operations.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionCmmaMatmul {
    pub matmul: CmmaMatmul,
    pub num_planes: u32,
    pub inner_layout: InnerLayout,
}

impl AttentionCmmaMatmul {
    fn bounce_config(&self, tile_shape: (u32, u32)) -> BounceConfig {
        BounceConfig {
            tile_shape,
            num_planes: self.num_planes,
            plane_dim: self.matmul.plane_dim,
            inner_layout: self.inner_layout,
        }
    }

    fn bounce_config_acc(&self) -> BounceConfig {
        self.bounce_config((self.matmul.tile_size.m, self.matmul.tile_size.n))
    }

    fn bounce_config_lhs(&self) -> BounceConfig {
        self.bounce_config((self.matmul.tile_size.m, self.matmul.tile_size.k))
    }
}

/// Attention's tile-level matmul configuration. Each variant carries the per-kind
/// config from cubek-std.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum AttentionTileMatmul {
    Cmma(AttentionCmmaMatmul),
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

    pub fn new_cmma(
        tile_size: TileSize,
        plane_dim: u32,
        num_planes: u32,
        inner_layout: InnerLayout,
    ) -> Self {
        AttentionTileMatmul::Cmma(AttentionCmmaMatmul {
            matmul: CmmaMatmul {
                tile_size,
                plane_dim,
                swizzle_modes: SwizzleModes::default(),
            },
            num_planes,
            inner_layout,
        })
    }

    pub fn tile_size(&self) -> TileSize {
        match self {
            AttentionTileMatmul::Cmma(c) => c.matmul.tile_size,
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
            cmma_allocate_lhs::<L, Plane>(MatrixLayout::RowMajor, c.matmul.tile_size)
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
            cmma_allocate_rhs::<R, Plane>(MatrixLayout::RowMajor, c.matmul.tile_size)
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
            cmma_allocate_rhs::<R, Plane>(MatrixLayout::ColMajor, c.matmul.tile_size)
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_rhs::<R, Plane>(MatrixLayout::ColMajor, c)
        }
    }
}

/// Allocates an accumulator tile that can be softmax'd (score) or scaled by
/// softmax stats (output). For the cmma path this is a `Tile::Bounce`
/// (cmma + smem + LocalTile) so row-wise ops can read/write through smem;
/// for the register path it falls back to `Tile::Register`.
#[cube]
pub fn allocate_rowwise_acc<A: Float>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<A, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            let matrix = unsafe {
                cubecl::cmma::Matrix::<A>::uninitialized(
                    cubecl::cmma::MatrixIdent::Accumulator,
                    c.matmul.tile_size.m as usize,
                    c.matmul.tile_size.n as usize,
                    c.matmul.tile_size.k as usize,
                    cubecl::cmma::MatrixLayout::Undefined,
                )
            };
            let cmma = CmmaTile::<A> {
                matrix,
                matrix_layout: MatrixLayout::RowMajor,
                tile_size: c.matmul.tile_size,
            };
            allocate_bounce_tile::<A, Plane>(cmma, c.bounce_config_acc())
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_acc::<A, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}

/// Allocates an LHS tile that receives the post-softmax cast-down values
/// (the value-matmul lhs). For the cmma path this is a `Tile::Bounce` so the
/// softmaxed values can be written through smem into the cmma fragment.
#[cube]
pub fn allocate_softmax_target_lhs<L: Float>(
    #[comptime] matmul: AttentionTileMatmul,
) -> Tile<L, Plane, ReadWrite> {
    match matmul {
        AttentionTileMatmul::Cmma(c) => {
            let matrix = unsafe {
                cubecl::cmma::Matrix::<L>::uninitialized(
                    cubecl::cmma::MatrixIdent::A,
                    c.matmul.tile_size.m as usize,
                    c.matmul.tile_size.n as usize,
                    c.matmul.tile_size.k as usize,
                    as_cmma_layout(MatrixLayout::RowMajor),
                )
            };
            let cmma = CmmaTile::<L> {
                matrix,
                matrix_layout: MatrixLayout::RowMajor,
                tile_size: c.matmul.tile_size,
            };
            allocate_bounce_tile::<L, Plane>(cmma, c.bounce_config_lhs())
        }
        AttentionTileMatmul::Register(c) => {
            register_allocate_lhs::<L, Plane>(MatrixLayout::RowMajor, c)
        }
    }
}
