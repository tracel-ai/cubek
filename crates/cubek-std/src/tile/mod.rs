pub mod cmma;
pub mod interleaved;
pub mod mma;
pub mod plane_vec_mat_inner_product;
pub mod register;
pub mod scope;

mod strided_tile;
mod tile_kind;

pub use cmma::*;
pub use interleaved::*;
pub use mma::*;
pub use plane_vec_mat_inner_product::*;
pub use register::*;
pub use scope::{Cube, Plane, Scope, ScopeMarker, Unit};
pub use strided_tile::*;
pub use tile_kind::*;

use std::marker::PhantomData;

use cubecl::{cmma::Matrix, prelude::*};

use crate::{MatrixLayout, StageIdent, tile::scope::Scope as TileScope};

#[derive(CubeType)]
pub enum Tile<N: Numeric, V: Size, Sc: TileScope, IO: SliceVisibility> {
    GlobalMemory(Slice<Vector<N, V>, IO>),
    SharedMemory(StridedTile<N, V, IO>),
    Cmma(CmmaTile<N>),
    MmaLhs(MmaLhsTile<N>),
    MmaRhs(MmaRhsTile<N>),
    MmaAcc(MmaAccTile<N>),
    Register(RegisterTile<N>),
    PlaneVec(PlaneVecTile<N, V>),
    Interleaved(InterleavedTile<N>),
    Broadcasted(Value<N>),
    None,
    _Phantom(ScopeMarker<Sc>),
}

#[derive(CubeType)]
pub struct CmmaTile<N: Numeric> {
    pub matrix: Matrix<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
}

#[derive(CubeType)]
pub struct MmaLhsTile<N: Numeric> {
    pub fragment: Array<Vector<N, mma::NL>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: MmaMatmul,
}

#[derive(CubeType)]
pub struct MmaRhsTile<N: Numeric> {
    pub fragment: Array<Vector<N, mma::NR>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: MmaMatmul,
}

#[derive(CubeType)]
pub struct MmaAccTile<N: Numeric> {
    pub fragment: Array<Vector<N, mma::NA>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: MmaMatmul,
}

#[derive(CubeType)]
pub struct RegisterTile<N: Numeric> {
    pub data: Array<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: RegisterMatmul,
}

#[derive(CubeType)]
pub struct PlaneVecTile<N: Numeric, V: Size> {
    // Fragment inner size is `NPlaneVec` (= reduce_vector_size), NOT the outer `V`.
    // V is retained only to satisfy the `Tile<N, V, Sc, IO>` enum variant type.
    pub data: Array<Vector<N, NPlaneVec>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: PlaneVecMatInnerProduct,
    #[cube(comptime)]
    pub _phantom_v: PhantomData<V>,
}

/// Wrapper over val to make enum work
#[derive(CubeType)]
pub struct Value<E: Numeric> {
    pub val: E,
}

#[cube]
impl<N: Numeric, V: Size, Sc: TileScope> Tile<N, V, Sc, ReadWrite> {
    /// Executes `lhs · rhs`, accumulating the result into `self`.
    pub fn mma<L: Numeric, VL: Size, R: Numeric, VR: Size>(
        &mut self,
        lhs: &Tile<L, VL, Sc, ReadWrite>,
        rhs: &Tile<R, VR, Sc, ReadWrite>,
    ) {
        match (lhs, rhs, self) {
            (Tile::Cmma(l), Tile::Cmma(r), Tile::Cmma(a)) => {
                cmma_execute(&l.matrix, &r.matrix, &mut a.matrix);
            }
            (Tile::MmaLhs(l), Tile::MmaRhs(r), Tile::MmaAcc(a)) => {
                mma_execute(
                    &l.fragment,
                    &r.fragment,
                    &mut a.fragment,
                    a.matrix_layout,
                    a.config,
                );
            }
            (Tile::Register(l), Tile::Register(r), Tile::Register(a)) => {
                register_execute(&l.data, &r.data, &mut a.data, a.config);
            }
            (Tile::PlaneVec(l), Tile::PlaneVec(r), Tile::PlaneVec(a)) => {
                planevec_execute(&l.data, &r.data, &mut a.data, a.config);
            }
            (Tile::Interleaved(l), Tile::Interleaved(r), Tile::Interleaved(a)) => {
                interleaved_execute(
                    &l.data,
                    l.matrix_layout,
                    &r.data,
                    r.matrix_layout,
                    &mut a.data,
                    a.matrix_layout,
                    a.config,
                );
            }
            _ => panic!("Unsupported storage combination for mma"),
        }
    }

    /// Copies data from `source` into `self`.
    pub fn copy_from<
        SE: Numeric,
        SS: Size,
        L: Numeric,
        R: Numeric,
        A: Numeric,
        SIO: SliceVisibility,
    >(
        &mut self,
        source: &Tile<SE, SS, Sc, SIO>,
        #[comptime] ident: StageIdent,
    ) {
        match (source, self) {
            // --- Cmma loads ---
            (Tile::SharedMemory(shared), Tile::Cmma(t)) => {
                cmma_load_from_shared::<SE, SS, N, V, SIO>(
                    shared,
                    &mut t.matrix,
                    ident,
                    t.matrix_layout,
                );
            }
            (Tile::None, Tile::Cmma(t)) => {
                cmma_load_zeros::<N, V>(&mut t.matrix);
            }

            // --- Mma loads ---
            (Tile::SharedMemory(shared), Tile::MmaLhs(t)) => {
                mma_load_lhs_from_shared::<SE, SS, N, R, A, SIO>(
                    shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                );
            }
            (Tile::SharedMemory(shared), Tile::MmaRhs(t)) => {
                mma_load_rhs_from_shared::<SE, SS, N, L, A, SIO>(
                    shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                );
            }
            (Tile::SharedMemory(shared), Tile::MmaAcc(t)) => {
                mma_load_acc_from_shared::<SE, SS, N, L, R, SIO>(
                    shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                );
            }
            (Tile::None, Tile::MmaAcc(t)) => {
                mma_load_acc_zeros::<SE, SS, N, L, R>(&mut t.fragment, t.matrix_layout, t.config);
            }

            // --- Register loads ---
            (Tile::SharedMemory(shared), Tile::Register(t)) => {
                register_load_from_shared::<SE, SS, N, V, SIO>(
                    shared,
                    &mut t.data,
                    t.matrix_layout,
                    t.config,
                    ident,
                );
            }
            (Tile::None, Tile::Register(t)) => {
                register_load_zeros::<N, V>(&mut t.data, t.config, ident);
            }

            // --- PlaneVec loads ---
            (Tile::SharedMemory(shared), Tile::PlaneVec(t)) => {
                planevec_load_from_shared::<SE, SS, N, SIO>(shared, &mut t.data, t.config, ident);
            }
            (Tile::None, Tile::PlaneVec(t)) => {
                planevec_load_zeros::<N>(&mut t.data, t.config);
            }

            // --- Interleaved loads ---
            (Tile::SharedMemory(shared), Tile::Interleaved(t)) => {
                interleaved_load_from_shared::<SE, SS, N, V, SIO>(
                    shared,
                    &mut t.data,
                    t.config,
                    ident,
                );
            }
            (Tile::None, Tile::Interleaved(t)) => {
                interleaved_load_zeros::<N, V>(&mut t.data, t.config);
            }

            // --- Writes: shared memory copies from a compute container ---
            (Tile::Cmma(t), Tile::SharedMemory(shared)) => {
                cmma_write_to_shared::<N, V, SE, SS>(shared, &t.matrix);
            }
            (Tile::MmaAcc(t), Tile::SharedMemory(shared)) => {
                mma_write_to_shared::<N, V, SE, L, R>(shared, &t.fragment, t.config);
            }
            (Tile::Register(t), Tile::SharedMemory(shared)) => {
                register_write_to_shared::<N, V, SE, SS>(shared, &t.data, t.config);
            }
            (Tile::PlaneVec(t), Tile::SharedMemory(shared)) => {
                planevec_write_to_shared::<SE, N, V>(shared, &t.data, t.config);
            }
            (Tile::Interleaved(t), Tile::SharedMemory(shared)) => {
                interleaved_write_to_shared::<N, V, SE, SS>(shared, &t.data, t.config);
            }

            _ => panic!("Unsupported storage pair for copy_from"),
        }
    }
}
