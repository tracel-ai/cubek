use std::marker::PhantomData;

use cubecl::{cmma::Matrix, prelude::*, std::tensor::layout::Coords2d};
use cubek_std::{MatrixLayout, tile::Strided};

use crate::components::tile::{
    NumericVector, VectorOf,
    cmma::{CmmaFragmentReader as _, CmmaStageReader},
};

pub struct Unit;
pub struct Plane;
pub struct Cube;

#[derive(CubeType)]
pub struct Tilex<N: NumericVector, S = Unit, IO: SliceVisibility = ReadOnly> {
    pub storage: TileStorage<N, IO>,
    pub layout: TileLayout,
    #[cube(comptime)]
    _scope: PhantomData<S>,
}

impl<N: NumericVector, S, IO: SliceVisibility> Tilex<N, S, IO> {
    pub fn cast<N2: NumericVector>(&self) -> Tilex<N2, IO> {
        let storage = match self.storage {
            TileStorage::Cmma(matrix) => {
                TileStorage::new_Cmma(cmma::cast::<N::Elem, N2::Elem>(&matrix))
            }
            _ => panic!("Unsupported"),
        };

        Tilex {
            storage,
            layout: self.layout.clone(),
            _scope: PhantomData,
        }
    }

    pub fn read(self) -> Tilex<N, S, ReadOnly> {
        Tilex {
            storage: self.storage.read(),
            layout: self.layout.clone(),
            _scope: PhantomData,
        }
    }

    pub fn write(self) -> Tilex<N, S, ReadWrite> {
        Tilex {
            storage: self.storage.write(),
            layout: self.layout.clone(),
            _scope: PhantomData,
        }
    }

    pub fn from_strided_tile() {
        // TODO
    }
}

#[cube]
pub fn tile_matmul<L: NumericVector, R: NumericVector, A: NumericVector>(
    lhs: &Tilex<L, Unit>,
    rhs: &Tilex<R, Unit>,
    acc: &mut Tilex<A, Unit, ReadWrite>,
) {
    match (&lhs.storage, &rhs.storage, &mut acc.storage) {
        (TileStorage::Cmma(lhs), TileStorage::Cmma(rhs), TileStorage::Cmma(acc)) => {
            cmma::execute::<L::Elem, R::Elem, A::Elem, A::Elem>(&lhs, &rhs, &acc, &acc)
        }
        _ => panic!("Unsupported"),
    }
}

#[cube]
pub fn tile_copy<F: NumericVector, T: NumericVector>(
    from: &Tilex<F>,
    to: &mut Tilex<T, ReadWrite>,
) {
    match (&from.storage, &mut to.storage) {
        (TileStorage::SharedMemory(strided_tile), TileStorage::Cmma(matrix)) => {
            CmmaStageReader::<Strided>::load_fragment(
                strided_tile,
                matrix,
                ComptimeOption::new_None(),
            )
        }
        _ => {}
    }
    // CmmaStageReader::<Strided>::load_fragment(from, to, ComptimeOption::new_None());
}

#[derive(CubeType)]
pub enum TileStorage<N: NumericVector, IO: SliceVisibility = ReadOnly> {
    GlobalMemory(Slice<VectorOf<N>, IO>),
    // SharedMemory(StridedTile<N::Elem, N::Size, IO>),
    SharedMemory(Slice<VectorOf<N>, IO>),
    LocalMemory(Slice<VectorOf<N>, IO>),
    Cmma(Matrix<N::Elem>),
    Mma(Array<VectorOf<N>>),
    Broadcasted(Value<N::Elem>),
}

impl<N: NumericVector, IO: SliceVisibility> TileStorage<N, IO> {
    pub fn read(self) -> TileStorage<N, ReadOnly> {
        match self {
            TileStorage::GlobalMemory(slice) => TileStorage::GlobalMemory(slice.to_slice()),
            TileStorage::SharedMemory(slice) => TileStorage::GlobalMemory(slice.to_slice()),
            TileStorage::LocalMemory(slice) => TileStorage::GlobalMemory(slice.to_slice()),
            TileStorage::Cmma(matrix) => todo!(),
            TileStorage::Mma(array) => todo!(),
            TileStorage::Broadcasted(value) => todo!(),
        }
    }

    pub fn write(self) -> TileStorage<N, ReadWrite> {
        match self {
            TileStorage::GlobalMemory(slice) => todo!(),
            TileStorage::SharedMemory(slice) => todo!(),
            TileStorage::LocalMemory(slice) => todo!(),
            TileStorage::Cmma(matrix) => todo!(),
            TileStorage::Mma(array) => todo!(),
            TileStorage::Broadcasted(value) => todo!(),
        }
    }
}

/// Wrapper over val to make enum work
#[derive(CubeType)]
pub struct Value<E: Numeric> {
    val: E,
}

#[derive(CubeType, Clone)]
pub enum TileLayout {
    Contiguous(MatrixLayout),
    Strided(StridedLayout),
}

#[derive(CubeType, Clone)]
pub struct StridedLayout {
    pub strides: Coords2d,
    pub shape: Coords2d,
}
