use cubecl::prelude::*;
use cubecl::std::CubeOption;
use std::marker::PhantomData;

use crate::components::tile::interleaved::config::InterleavedMatmulConfig;
use crate::components::tile::{
    StridedTile,
    io::{Filled, Strided, TileKind},
    register::{RegisterMatmul, UnitFragment},
};
use crate::definition::StageIdent;

/// Reader for the register matmul fragments. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct InterleavedStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

/// Generic register reader over any tile kind
#[cube]
pub(super) trait InterleavedFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &<Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: InterleavedMatmulConfig,
    );
}

#[cube]
impl InterleavedFragmentReader for InterleavedStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        frag: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
    }
}

#[cube]
impl InterleavedFragmentReader for InterleavedStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric>(
        value: &V,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
    }
}

#[cube]
impl<Inner: TileKind> InterleavedFragmentReader for InterleavedStageReader<CubeOption<Inner>>
where
    InterleavedStageReader<Inner>: InterleavedFragmentReader<TileKind = Inner>,
{
    type TileKind = CubeOption<Inner>;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &CubeOption<Inner::Tile<V>>,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
    }
}
