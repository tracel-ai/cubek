use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout,
    tile::{Filled, Strided, StridedTile, Tile, TileKind},
};

use crate::components::tile::plane_vec_mat_inner_product::VectorContainer;

/// Reader for the vector side of the VecMat operation
#[derive(CubeType)]
pub struct VectorStageReader {}

/// Generic matrix reader over any tile type
#[cube]
pub(super) trait MatrixFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &Tile<Self::TileKind, V, N>,
        frag: &mut Sequence<VectorContainer<E>>,
        #[comptime] n: u32,
    );
}

/// Reader for the matrix side of the VecMat operation. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct MatrixStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl VectorStageReader {
    pub fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &StridedTile<V, N>,
        frag: &mut VectorContainer<E>,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::RowMajor));

        let offset = tile.stage_offset(UNIT_POS_X);
        frag.vector = Vector::cast_from(tile.stage[offset as usize]);
    }
}

#[cube]
impl MatrixFragmentReader for MatrixStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &StridedTile<V, N>,
        frag: &mut Sequence<VectorContainer<E>>,
        #[comptime] n: u32,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::ColMajor));

        #[unroll]
        for n in 0..n {
            let vector_container = frag.index_mut(n as usize);
            let offset = tile.stage_offset(UNIT_POS_X + n * tile.stride);
            vector_container.vector = Vector::cast_from(tile.stage[offset as usize]);
        }
    }
}

#[cube]
impl MatrixFragmentReader for MatrixStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        value: &V,
        frag: &mut Sequence<VectorContainer<E>>,
        #[comptime] n: u32,
    ) {
        #[unroll]
        for n in 0..n as usize {
            let vector_container = frag.index_mut(n);
            vector_container.vector = Vector::cast_from(*value);
        }
    }
}

#[cube]
impl<Inner: TileKind> MatrixFragmentReader for MatrixStageReader<Option<Inner>>
where
    MatrixStageReader<Inner>: MatrixFragmentReader<TileKind = Inner>,
{
    type TileKind = Option<Inner>;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &ComptimeOption<Inner::Tile<V, N>>,
        frag: &mut Sequence<VectorContainer<E>>,
        #[comptime] n: u32,
    ) {
        #[comptime]
        #[comptime]
        match tile {
            ComptimeOption::Some(tile) => MatrixStageReader::<Inner>::load_fragment(tile, frag, n),
            ComptimeOption::None => {
                MatrixStageReader::<Filled>::load_fragment::<E, V, N>(&V::from_int(0), frag, n)
            }
        }
    }
}
