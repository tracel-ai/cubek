use std::marker::PhantomData;

use cubecl::{cmma::Matrix, prelude::*, std::tensor::layout::Coords2d};
use cubek_std::{
    MatrixLayout,
    tile::{Strided, StridedTile},
};

use crate::components::tile::cmma::{CmmaFragmentReader as _, CmmaStageReader};

// TODO
// pub struct Unit;
pub struct Plane;
// pub struct Cube;

#[derive(CubeType)]
pub struct Tilex<N: Numeric, V: Size, Scope, IO: SliceVisibility> {
    pub storage: TileStorage<N, V, IO>,
    pub layout: TileLayout,
    #[cube(comptime)]
    pub _scope: PhantomData<Scope>,
}

#[cube]
impl<N: Numeric, V: Size, S, IO: SliceVisibility> Tilex<N, V, S, IO> {
    pub fn with_elem<N2: Numeric>(&self) -> Tilex<N2, V, S, IO> {
        todo!()
        // let storage = match self.storage {
        //     TileStorage::Cmma(matrix) => TileStorage::new_Cmma(cmma::cast::<N, N2>(&matrix)),
        //     _ => panic!("Unsupported"),
        // };

        // Tilex::<N2, V, S, IO> {
        //     storage,
        //     layout: self.layout.clone(),
        //     _scope: PhantomData,
        // }
    }

    pub fn with_size<V2: Size>(&self) -> Tilex<N, V2, S, IO> {
        todo!()
        // let storage = match self.storage {
        //     _ => panic!("Unsupported"),
        // };

        // Tilex::<N, V2, S, IO> {
        //     storage,
        //     layout: self.layout.clone(),
        //     _scope: PhantomData,
        // }
    }

    pub fn with_scope<S2>(&self) -> Tilex<N, V, S2, IO> {
        todo!()
        // let storage = match self.storage {
        //     _ => panic!("Unsupported"),
        // };

        // Tilex::<N, V, S2, IO> {
        //     storage,
        //     layout: self.layout.clone(),
        //     _scope: PhantomData,
        // }
    }

    // pub fn read(self) -> Tilex<N, V, S, ReadOnly> {
    //     Tilex {
    //         storage: self.storage.read(),
    //         layout: self.layout.clone(),
    //         _scope: PhantomData,
    //     }
    // }

    // pub fn write(self) -> Tilex<N, V, S, ReadWrite> {
    //     Tilex {
    //         storage: self.storage.write(),
    //         layout: self.layout.clone(),
    //         _scope: PhantomData,
    //     }
    // }

    // pub fn from_strided_tile() {
    //     // TODO
    // }
}

#[cube]
pub fn tile_matmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size, Scope>(
    lhs: &Tilex<L, VL, Scope, ReadOnly>,
    rhs: &Tilex<R, VR, Scope, ReadOnly>,
    acc: &mut Tilex<A, VA, Scope, ReadWrite>,
) {
    match (&lhs.storage, &rhs.storage, &mut acc.storage) {
        (TileStorage::Cmma(lhs), TileStorage::Cmma(rhs), TileStorage::Cmma(acc)) => {
            cmma::execute::<L, R, A, A>(&lhs, &rhs, &acc, &acc)
        }
        _ => panic!("Unsupported"),
    }
}

#[cube]
pub fn tile_copy<F: Numeric, VF: Size, T: Numeric, VT: Size, Scope>(
    from: &Tilex<F, VF, Scope, ReadOnly>,
    to: &mut Tilex<T, VT, Scope, ReadWrite>,
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
pub enum TileStorage<N: Numeric, VN: Size, IO: SliceVisibility> {
    GlobalMemory(Slice<Vector<N, VN>, IO>),
    SharedMemory(StridedTile<N, VN, IO>),
    LocalMemory(Slice<Vector<N, VN>, IO>),
    Cmma(Matrix<N>),
    Mma(Array<Vector<N, VN>>),
    Broadcasted(Value<N>),
    None,
}

// impl<N: Numeric, VN: Size> TileStorage<N, VN, ReadWrite> {
//     pub fn read(self) -> TileStorage<N, VN, ReadOnly> {
//         match self {
//             TileStorage::GlobalMemory(slice) => TileStorage::GlobalMemory(slice.to_slice()),
//             TileStorage::SharedMemory(strided_tile) => {
//                 TileStorage::SharedMemory(strided_tile.as_slice())
//             }
//             TileStorage::LocalMemory(slice) => TileStorage::LocalMemory(slice.to_slice()),
//             TileStorage::Cmma(matrix) => todo!(),
//             TileStorage::Mma(array) => todo!(),
//             TileStorage::Broadcasted(value) => todo!(),
//         }
//     }
// }

// impl<N: Numeric, VN: Size> TileStorage<N, VN, ReadOnly> {
//     pub fn write(self) -> TileStorage<N, VN, ReadWrite> {
//         match self {
//             TileStorage::GlobalMemory(slice) => todo!(),
//             TileStorage::SharedMemory(slice) => todo!(),
//             TileStorage::LocalMemory(slice) => todo!(),
//             TileStorage::Cmma(matrix) => todo!(),
//             TileStorage::Mma(array) => todo!(),
//             TileStorage::Broadcasted(value) => todo!(),
//         }
//     }
// }

/// Wrapper over val to make enum work
#[derive(CubeType)]
pub struct Value<E: Numeric> {
    pub val: E,
}

#[derive(CubeType, Clone)]
pub enum TileLayout {
    Contiguous(ContiguousMatrixLayout),
    Strided(StridedLayout),
    None,
}

#[cube]
impl TileLayout {
    pub fn new_contiguous(#[comptime] layout: MatrixLayout) -> TileLayout {
        TileLayout::new_Contiguous(ContiguousMatrixLayout { layout })
    }
}

#[derive(CubeType, Clone)]
pub struct StridedLayout {
    pub strides: Coords2d,
    pub shape: Coords2d,
}

#[derive(CubeType, Clone)]
/// Wrapper over comptime so it can fit in cube enum
pub struct ContiguousMatrixLayout {
    #[cube(comptime)]
    pub layout: MatrixLayout,
}
