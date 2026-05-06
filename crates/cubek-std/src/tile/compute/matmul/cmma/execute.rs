use cubecl::{cmma, prelude::*};

use crate::{
    MatrixLayout, StageIdent, as_cmma_layout,
    tile::{
        compute::matmul::cmma::{CmmaFragmentReader as _, CmmaStageReader, CmmaStageWriter},
        data::{Strided, StridedTile},
    },
};

#[cube]
pub fn cmma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &cmma::Matrix<L>,
    rhs: &cmma::Matrix<R>,
    acc: &mut cmma::Matrix<A>,
) {
    cmma::execute::<L, R, A, A>(lhs, rhs, acc, acc);
}

#[cube]
pub fn cmma_load_from_shared<E: Numeric, ES: Size, N: Numeric, IO: SliceVisibility>(
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
pub fn cmma_load_zeros<N: Numeric>(matrix: &mut cmma::Matrix<N>) {
    cmma::fill(matrix, N::from_int(0));
}

#[cube]
pub fn cmma_write_to_shared<E: Numeric, ES: Size, A: Numeric>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    matrix: &cmma::Matrix<A>,
) {
    let casted = cmma::cast::<A, E>(matrix);
    CmmaStageWriter::store_fragment(shared, &casted);
}
