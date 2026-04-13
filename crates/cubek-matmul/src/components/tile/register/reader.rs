use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout,
    tile::{Filled, Strided, StridedTile, TileKind},
};
use std::marker::PhantomData;

use crate::components::tile::register::{
    RegisterMatmul, UnitFragment,
    config::{ProductType, RegisterMatmulConfig},
};
use crate::definition::StageIdent;

/// Reader for the register matmul fragments. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct RegisterStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

/// Generic register reader over any tile kind
#[cube]
pub(super) trait RegisterFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &<Self::TileKind as TileKind>::Tile<V, N>,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    );
}

#[cube]
impl RegisterFragmentReader for RegisterStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &StridedTile<V, N>,
        frag: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        // Could these be unified somehow?
        match ident {
            StageIdent::Lhs => load_lhs(tile, frag, config),
            StageIdent::Rhs => load_rhs(tile, frag, config),
            StageIdent::Acc => load_acc(tile, frag, config),
            StageIdent::Out => panic!("Can't load out"),
        }
    }
}

#[cube]
fn load_lhs<E: Numeric, V: Numeric, N: Size>(
    tile: &StridedTile<V, N>,
    frag: &mut UnitFragment<E>,
    #[comptime] config: RegisterMatmulConfig,
) {
    let size = config.shared.tile_size;

    match config.product_type {
        ProductType::Inner => match frag.layout.comptime() {
            MatrixLayout::RowMajor => {
                RegisterMatmul::load_plain(tile, &mut frag.array, size.m(), size.k());
            }
            MatrixLayout::ColMajor => {
                RegisterMatmul::load_transposed(tile, &mut frag.array, size.k(), size.m());
            }
        },
        ProductType::Outer => match frag.layout.comptime() {
            MatrixLayout::RowMajor => {
                RegisterMatmul::load_transposed(tile, &mut frag.array, size.m(), size.k());
            }
            MatrixLayout::ColMajor => {
                RegisterMatmul::load_plain(tile, &mut frag.array, size.k(), size.m());
            }
        },
    }
}

#[cube]
fn load_rhs<E: Numeric, V: Numeric, N: Size>(
    tile: &StridedTile<V, N>,
    frag: &mut UnitFragment<E>,
    #[comptime] config: RegisterMatmulConfig,
) {
    let size = config.shared.tile_size;

    match config.product_type {
        ProductType::Inner => match frag.layout.comptime() {
            MatrixLayout::RowMajor => {
                RegisterMatmul::load_transposed(tile, &mut frag.array, size.k(), size.n());
            }
            MatrixLayout::ColMajor => {
                RegisterMatmul::load_plain(tile, &mut frag.array, size.n(), size.k());
            }
        },
        ProductType::Outer => match frag.layout.comptime() {
            MatrixLayout::RowMajor => {
                RegisterMatmul::load_plain(tile, &mut frag.array, size.k(), size.n());
            }
            MatrixLayout::ColMajor => {
                RegisterMatmul::load_transposed(tile, &mut frag.array, size.n(), size.k());
            }
        },
    }
}

#[cube]
fn load_acc<E: Numeric, V: Numeric, N: Size>(
    tile: &StridedTile<V, N>,
    frag: &mut UnitFragment<E>,
    #[comptime] config: RegisterMatmulConfig,
) {
    let size = config.shared.tile_size;

    match frag.layout.comptime() {
        MatrixLayout::RowMajor => {
            RegisterMatmul::load_plain(tile, &mut frag.array, size.m(), size.n());
        }
        MatrixLayout::ColMajor => {
            RegisterMatmul::load_transposed(tile, &mut frag.array, size.n(), size.m());
        }
    }
}

#[cube]
impl RegisterFragmentReader for RegisterStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        value: &V,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        let size = config.shared.tile_size;
        let size = match ident {
            StageIdent::Lhs => size.mk(),
            StageIdent::Rhs => size.nk(),
            StageIdent::Acc => size.mn(),
            StageIdent::Out => size.mn(),
        };

        for i in 0..size {
            fragment.array[i as usize] = E::cast_from(*value);
        }
    }
}

#[cube]
impl<Inner: TileKind> RegisterFragmentReader for RegisterStageReader<Option<Inner>>
where
    RegisterStageReader<Inner>: RegisterFragmentReader<TileKind = Inner>,
{
    type TileKind = Option<Inner>;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &ComptimeOption<Inner::Tile<V, N>>,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        #[comptime]
        #[comptime]
        match tile {
            ComptimeOption::Some(tile) => {
                RegisterStageReader::<Inner>::load_fragment(tile, fragment, ident, config)
            }
            ComptimeOption::None => RegisterStageReader::<Filled>::load_fragment::<E, V, N>(
                &V::from_int(0),
                fragment,
                ident,
                config,
            ),
        }
    }
}
