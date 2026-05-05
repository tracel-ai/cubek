use cubecl::prelude::*;

use crate::{
    MatrixLayout, SwizzleModes, TileSize,
    tile::{Tile, TileScope},
};

#[derive(CubeType)]
pub struct InterleavedTile<N: Numeric> {
    pub data: Array<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: InterleavedMatmul,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InterleavedMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl InterleavedMatmul {
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

#[cube]
pub fn interleaved_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: InterleavedMatmul,
) -> Tile<L, Sc, ReadWrite> {
    let m = config.tile_size.m();
    let k = config.tile_size.k();
    let plane_dim = config.plane_dim;
    Tile::new_Interleaved(InterleavedTile::<L> {
        data: Array::new((m * (k / plane_dim)) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn interleaved_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: InterleavedMatmul,
) -> Tile<R, Sc, ReadWrite> {
    let n = config.tile_size.n();
    let k = config.tile_size.k();
    let plane_dim = config.plane_dim;
    Tile::new_Interleaved(InterleavedTile::<R> {
        data: Array::new(((k / plane_dim) * n) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn interleaved_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: InterleavedMatmul,
) -> Tile<A, Sc, ReadWrite> {
    let m = config.tile_size.m();
    let n = config.tile_size.n();
    Tile::new_Interleaved(InterleavedTile::<A> {
        data: Array::new((m * n) as usize),
        matrix_layout: layout,
        config,
    })
}
