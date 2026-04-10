use cubecl::{cmma::Matrix, prelude::*, std::tensor::layout::Coords2d};
use cubek_std::MatrixLayout;

pub type TileScalar<E, IO = ReadOnly> = Tilex<E, Const<1>, IO>;

#[derive(CubeType)]
pub struct Tilex<E: Scalar, N: Size, IO: SliceVisibility = ReadOnly> {
    pub storage: TileStorage<E, N, IO>,
    pub layout: TileLayout,
}

#[cube]
pub fn tile_matmul<L: Scalar, R: Scalar, A: Scalar, NL: Size, NR: Size, NA: Size>(
    lhs: &Tilex<L, NL>,
    rhs: &Tilex<R, NR>,
    acc: &mut Tilex<A, NA>,
) {
    match (&lhs.storage, &rhs.storage, &mut acc.storage) {
        (TileStorage::Cmma(lhs), TileStorage::Cmma(rhs), TileStorage::Cmma(acc)) => {
            cmma::execute::<L, R, A, A>(&lhs, &rhs, &acc, &acc)
        }
        _ => panic!("Unsupported"),
    }
}

#[derive(CubeType)]
pub enum TileStorage<E: Scalar, N: Size, IO: SliceVisibility = ReadOnly> {
    GlobalMemory(Slice<Vector<E, N>, IO>),
    SharedMemory(Slice<Vector<E, N>, IO>),
    LocalMemory(Slice<Vector<E, N>, IO>),
    Cmma(Matrix<E>),
    Mma(Array<Vector<E, N>>),
    Broadcasted(E),
}

#[derive(CubeType)]
pub enum TileLayout {
    Contiguous(MatrixLayout),
    Strided(StridedLayout),
}

#[derive(CubeType)]
pub struct StridedLayout {
    pub strides: Coords2d,
    pub shape: Coords2d,
}

#[cube]
impl<E: Scalar, N: Size, IO: SliceVisibility> TileStorage<E, N, IO> {
    pub fn as_slice(&self) -> Slice<Vector<E, N>, IO> {
        match &self {
            TileStorageKind::GlobalMemory(slice) => slice.clone(),
            TileStorageKind::SharedMemory(slice) => slice.clone(),
            TileStorageKind::LocalMemory(slice) => slice.clone(),
            TileStorageKind::Cmma(_) => panic!(),
            TileStorageKind::Mma(_) => panic!(),
            TileStorageKind::Broadcasted(_) => panic!(),
        }
    }
}
