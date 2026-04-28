use cubecl::{
    cmma::{self},
    prelude::*,
};

use crate::{
    MatrixLayout, StageIdent, SwizzleModes, TileSize, as_cmma_layout,
    tile::{
        CmmaFragmentReader, CmmaStageReader, CmmaStageWriter, CmmaTile, Strided, StridedTile, Tile,
        scope::Scope,
    },
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CmmaMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl CmmaMatmul {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_modes: SwizzleModes) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
        }
    }
}

#[cube]
pub fn cmma_allocate_lhs<L: Numeric, VL: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<L, VL, Sc, ReadWrite> {
    let fragment = unsafe {
        cmma::Matrix::<L>::uninitialized(
            cmma::MatrixIdent::A,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            as_cmma_layout(layout),
        )
    };
    Tile::new_Cmma(CmmaTile::<L> {
        matrix: fragment,
        matrix_layout: layout,
    })
}

#[cube]
pub fn cmma_allocate_rhs<R: Numeric, VR: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<R, VR, Sc, ReadWrite> {
    let fragment = unsafe {
        cmma::Matrix::<R>::uninitialized(
            cmma::MatrixIdent::B,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            as_cmma_layout(layout),
        )
    };
    Tile::new_Cmma(CmmaTile::<R> {
        matrix: fragment,
        matrix_layout: layout,
    })
}

#[cube]
pub fn cmma_allocate_acc<A: Numeric, VA: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<A, VA, Sc, ReadWrite> {
    let fragment = unsafe {
        cmma::Matrix::<A>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            cmma::MatrixLayout::Undefined,
        )
    };
    Tile::new_Cmma(CmmaTile::<A> {
        matrix: fragment,
        matrix_layout: layout,
    })
}

#[cube]
pub fn cmma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &cmma::Matrix<L>,
    rhs: &cmma::Matrix<R>,
    acc: &mut cmma::Matrix<A>,
) {
    cmma::execute::<L, R, A, A>(lhs, rhs, acc, acc);
}

#[cube]
pub fn cmma_load_from_shared<E: Numeric, ES: Size, N: Numeric, V: Size, IO: SliceVisibility>(
    shared: &StridedTile<E, ES, IO>,
    matrix: &mut cmma::Matrix<N>,
    #[comptime] ident: StageIdent,
    #[comptime] matrix_layout: MatrixLayout,
) {
    let shared = shared.to_read_only();
    match ident {
        StageIdent::Lhs | StageIdent::Rhs => {
            CmmaStageReader::<Strided>::load_fragment(&shared, matrix, ComptimeOption::new_None());
        }
        StageIdent::Acc => {
            CmmaStageReader::<Strided>::load_fragment(
                &shared,
                matrix,
                ComptimeOption::new_Some(as_cmma_layout(matrix_layout)),
            );
        }
        _ => panic!("Invalid ident for CMMA load"),
    }
}

#[cube]
pub fn cmma_load_zeros<N: Numeric, V: Size>(matrix: &mut cmma::Matrix<N>) {
    cmma::fill(matrix, N::from_int(0));
}

#[cube]
pub fn cmma_write_to_shared<E: Numeric, ES: Size, A: Numeric, VA: Size>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    matrix: &cmma::Matrix<A>,
) {
    let casted = cmma::cast::<A, E>(matrix);
    CmmaStageWriter::store_fragment(shared, &casted);
}
