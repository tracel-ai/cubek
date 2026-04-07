use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::matmul::InnerMatmul,
    components::tile::pipeline::{UnitTile, UnitTileLayout},
};

use cubek_std::{TileSize, tile::StridedTile};

#[derive(CubeType)]
pub struct UnitMatmul<A: Numeric, B: Numeric, CD: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<(A, B, CD)>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitMatmulConfig {
    pub tile_size: TileSize,
}

#[cube]
impl<A: Numeric, B: Numeric, CD: Numeric> InnerMatmul for UnitMatmul<A, B, CD> {
    type Lhs = UnitTile<A>;
    type Rhs = UnitTile<B>;
    type Acc = UnitTile<CD>;
    type Config = UnitMatmulConfig;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        UnitTile::new(UnitTileLayout::new(
            config.tile_size.m,
            config.tile_size.k,
            false,
        ))
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        UnitTile::new(UnitTileLayout::new(
            config.tile_size.k,
            config.tile_size.n,
            false,
        ))
    }

    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Self::Rhs {
        UnitTile::new(UnitTileLayout::new(
            config.tile_size.k,
            config.tile_size.n,
            true,
        ))
    }

    fn load_lhs<E: Numeric, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Lhs) {
        fragment.load_from_strided_tile(tile);
    }

    fn load_rhs<E: Float, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Rhs) {
        fragment.load_from_strided_tile(tile);
    }

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Acc,
        #[comptime] tile_size: TileSize,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = tile_size.into(); (m, n, k)};
        for m_ in 0..m {
            for n_ in 0..n {
                let mut sum = CD::from_int(0);
                for k_ in 0..k {
                    let lhs_val = lhs.get(m_, k_);
                    let rhs_val = rhs.get(k_, n_);
                    sum += CD::cast_from(lhs_val) * CD::cast_from(rhs_val);
                }
                out.accumulate(m_, n_, sum);
            }
        }
    }
}
