use std::marker::PhantomData;

use cubecl::prelude::*;

use cubek_std::tile::{Filled, Strided, StridedTile, TileKind};

/// Generic CMMA reader over any tile type
#[cube]
pub(crate) trait CmmaFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &<Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut cmma::Matrix<E>,
        layout: Option<cmma::MatrixLayout>,
    );
}

/// Reader using the cmma load/fill functions. Tile kind determines implementation.
#[derive(CubeType)]
pub struct CmmaStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl CmmaFragmentReader for CmmaStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        fragment: &mut cmma::Matrix<E>,
        layout: Option<cmma::MatrixLayout>,
    ) {
        let (slice, stride) = tile.as_unlined();
        match layout {
            None => cmma::load(fragment, &slice, stride),
            Some(layout) => cmma::load_with_layout(fragment, &slice, stride, layout),
        }
    }
}

#[cube]
impl CmmaFragmentReader for CmmaStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric>(
        value: &V,
        fragment: &mut cmma::Matrix<E>,
        _layout: Option<cmma::MatrixLayout>,
    ) {
        cmma::fill(fragment, E::cast_from(*value));
    }
}

#[cube]
impl<Inner: TileKind> CmmaFragmentReader for CmmaStageReader<Option<Inner>>
where
    CmmaStageReader<Inner>: CmmaFragmentReader<TileKind = Inner>,
{
    type TileKind = Option<Inner>;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &Option<Inner::Tile<V>>,
        fragment: &mut cmma::Matrix<E>,
        layout: Option<cmma::MatrixLayout>,
    ) {
        match tile {
            Some(tile) => CmmaStageReader::<Inner>::load_fragment(tile, fragment, layout),
            None => {
                CmmaStageReader::<Filled>::load_fragment::<E, V>(&V::from_int(0), fragment, layout)
            }
        }
    }
}
