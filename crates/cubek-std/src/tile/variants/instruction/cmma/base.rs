use cubecl;
use cubecl::{
    cmma::{self},
    prelude::*,
};

use crate::{
    MatrixLayout, StageIdent, TileSize, as_cmma_layout,
    tile::{
        SharedTile, Tile, TileKind, TileKindExpand, TileScope,
        variants::instruction::cmma::{CmmaStageWriter, cmma_load_strided},
    },
};

#[derive(CubeType)]
pub struct CmmaTile<N: Numeric> {
    pub matrix: cmma::Matrix<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub tile_size: TileSize,
}

#[cube]
impl<E: Float> CmmaTile<E> {
    pub fn fill_zero(&mut self) {
        cubecl::cmma::fill(&mut self.matrix, E::from_int(0));
    }
}

#[cube]
impl<A: Numeric> CmmaTile<A> {
    /// Executes `lhs · rhs`, accumulating into `self`.
    pub fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &CmmaTile<L>, rhs: &CmmaTile<R>) {
        cmma_execute(&lhs.matrix, &rhs.matrix, &mut self.matrix);
    }
}

#[cube]
impl<N: Numeric> CmmaTile<N> {
    /// Supported sources: `SharedTile` (load) and `None` (zero-init).
    pub fn copy_from<SE: Numeric, SS: Size, Sc: TileScope>(
        &mut self,
        source: &Tile<SE, Sc>,
        #[comptime] ident: StageIdent,
    ) {
        match &source.kind {
            TileKind::SharedTile(shared) => {
                cmma_load_from_shared::<SE, SS, N>(
                    shared,
                    &mut self.matrix,
                    ident,
                    self.matrix_layout,
                );
            }
            TileKind::None => cmma_load_zeros::<N>(&mut self.matrix),
            TileKind::Cmma(_)
            | TileKind::Mma(_)
            | TileKind::Register(_)
            | TileKind::PlaneVec(_)
            | TileKind::Interleaved(_)
            | TileKind::Unit(_)
            | TileKind::WhiteboxFragment(_)
            | TileKind::RowWise(_)
            | TileKind::Bounce(_)
            | TileKind::Stage(_)
            | TileKind::Partition(_)
            | TileKind::Pipelined(_) => panic!("CmmaTile::copy_from: unsupported source variant"),
        }
    }

    pub fn init_zero(&mut self) {
        cmma_load_zeros::<N>(&mut self.matrix);
    }
}

#[cube]
pub fn cmma_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<L, Sc> {
    let fragment = unsafe {
        cmma::Matrix::<L>::uninitialized(
            cmma::MatrixIdent::A,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            as_cmma_layout(layout),
        )
    };
    Tile::from_kind(TileKind::new_Cmma(CmmaTile::<L> {
        matrix: fragment,
        matrix_layout: layout,
        tile_size,
    }))
}

#[cube]
pub fn cmma_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<R, Sc> {
    let fragment = unsafe {
        cmma::Matrix::<R>::uninitialized(
            cmma::MatrixIdent::B,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            as_cmma_layout(layout),
        )
    };
    Tile::from_kind(TileKind::new_Cmma(CmmaTile::<R> {
        matrix: fragment,
        matrix_layout: layout,
        tile_size,
    }))
}

#[cube]
pub fn cmma_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<A, Sc> {
    let fragment = unsafe {
        cmma::Matrix::<A>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            cmma::MatrixLayout::Undefined,
        )
    };
    Tile::from_kind(TileKind::new_Cmma(CmmaTile::<A> {
        matrix: fragment,
        matrix_layout: layout,
        tile_size,
    }))
}

// ===========================================================================
// Compute: matmul / load / write / zero-init
// ===========================================================================

#[cube]
pub fn cmma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &cmma::Matrix<L>,
    rhs: &cmma::Matrix<R>,
    acc: &mut cmma::Matrix<A>,
) {
    cmma::execute(lhs, rhs, &*acc, &*acc);
}

#[cube]
pub fn cmma_load_from_shared<E: Numeric, ES: Size, N: Numeric>(
    shared: &SharedTile<E>,
    matrix: &mut cmma::Matrix<N>,
    #[comptime] ident: StageIdent,
    #[comptime] matrix_layout: MatrixLayout,
) {
    let shared = shared.view::<ES>();
    match ident {
        StageIdent::Lhs | StageIdent::Rhs => {
            cmma_load_strided(&shared, matrix, ComptimeOption::new_None());
        }
        StageIdent::Acc => {
            cmma_load_strided(
                &shared,
                matrix,
                ComptimeOption::new_Some(as_cmma_layout(matrix_layout)),
            );
        }
        _ => panic!("Invalid ident for CMMA load"),
    }
}

#[cube]
pub fn cmma_load_zeros<N: Numeric>(matrix: &mut cmma::Matrix<N>) {
    cmma::fill(matrix, N::from_int(0));
}

#[cube]
pub fn cmma_write_to_shared<E: Numeric, ES: Size, A: Numeric>(
    shared: &mut SharedTile<E>,
    matrix: &cmma::Matrix<A>,
) {
    let mut shared = shared.view::<ES>();
    let casted: cmma::Matrix<E> = cmma::cast(matrix);
    CmmaStageWriter::store_fragment(&mut shared, &casted);
}
