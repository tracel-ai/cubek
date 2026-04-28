use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::tile::{Filled, Strided, StridedTile, TileKind};

/// Generic CMMA reader over any tile type
#[cube]
pub trait CmmaFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &<Self::TileKind as TileKind>::Tile<V, N>,
        fragment: &mut cmma::Matrix<E>,
        layout: ComptimeOption<cmma::MatrixLayout>,
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

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &StridedTile<V, N>,
        fragment: &mut cmma::Matrix<E>,
        layout: ComptimeOption<cmma::MatrixLayout>,
    ) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        #[comptime]
        match layout {
            ComptimeOption::None => cmma::load(fragment, &slice, stride),
            ComptimeOption::Some(layout) => {
                cmma::load_with_layout(fragment, &slice, stride, layout)
            }
        }
    }
}

#[cube]
impl CmmaFragmentReader for CmmaStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        value: &V,
        fragment: &mut cmma::Matrix<E>,
        _layout: ComptimeOption<cmma::MatrixLayout>,
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

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &ComptimeOption<Inner::Tile<V, N>>,
        fragment: &mut cmma::Matrix<E>,
        layout: ComptimeOption<cmma::MatrixLayout>,
    ) {
        #[comptime]
        #[comptime]
        match tile {
            ComptimeOption::Some(tile) => {
                CmmaStageReader::<Inner>::load_fragment(tile, fragment, layout)
            }
            ComptimeOption::None => CmmaStageReader::<Filled>::load_fragment::<E, V, N>(
                &V::from_int(0),
                fragment,
                layout,
            ),
        }
    }
}
