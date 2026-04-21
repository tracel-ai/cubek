use std::marker::PhantomData;

use cubecl::{cmma::Matrix, prelude::*};
use cubek_std::{
    MatrixLayout,
    tile::{StridedTile, mma::MmaIOConfig},
};

use crate::components::tile_matmul::{ProductType, SharedTileConfig};
use crate::definition::StageIdent;

pub mod cmma;
pub mod interleaved;
pub mod mma;
pub mod planevec;
pub mod register;

pub use cmma::*;
pub use interleaved::*;
pub use mma::*;
pub use planevec::*;
pub use register::*;

/// Identifies which compute primitive executes a tile matmul.
pub trait Scope: Clone + Copy + Send + Sync + 'static {}

#[derive(Clone, Copy)]
pub struct Unit;
#[derive(Clone, Copy)]
pub struct Plane;
#[derive(Clone, Copy)]
pub struct Cube;

impl Scope for Unit {}
impl Scope for Plane {}
impl Scope for Cube {}

/// Zero-sized comptime marker used to carry a [Scope] generic through [Tile].
#[derive(CubeType, Clone, Copy)]
pub struct ScopeMarker<Sc: Scope> {
    #[cube(comptime)]
    _phantom: PhantomData<Sc>,
}

#[derive(CubeType)]
pub enum Tile<N: Numeric, V: Size, Sc: Scope, IO: SliceVisibility> {
    GlobalMemory(Slice<Vector<N, V>, IO>),
    SharedMemory(StridedTile<N, V, IO>),
    Cmma(CmmaTile<N>),
    Mma(MmaTile<N, V>),
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
pub struct MmaTile<N: Numeric, V: Size> {
    pub fragment: Array<Vector<N, V>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: SharedTileConfig,
    #[cube(comptime)]
    pub mma_io_config: MmaIOConfig,
}

#[derive(CubeType)]
pub struct RegisterTile<N: Numeric> {
    pub data: Array<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: SharedTileConfig,
    #[cube(comptime)]
    pub product_type: ProductType,
}

#[derive(CubeType)]
pub struct PlaneVecTile<N: Numeric, V: Size> {
    pub data: Array<Vector<N, V>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: SharedTileConfig,
    #[cube(comptime)]
    pub reduce_vector_size: u32,
}

#[derive(CubeType)]
pub struct InterleavedTile<N: Numeric> {
    pub data: Array<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: SharedTileConfig,
}

/// Wrapper over val to make enum work
#[derive(CubeType)]
pub struct Value<E: Numeric> {
    pub val: E,
}

#[cube]
impl<N: Numeric, V: Size, Sc: Scope> Tile<N, V, Sc, ReadWrite> {
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
            (Tile::Mma(l), Tile::Mma(r), Tile::Mma(a)) => {
                mma_execute(
                    &l.fragment,
                    &r.fragment,
                    &mut a.fragment,
                    a.matrix_layout,
                    a.config,
                    a.mma_io_config,
                );
            }
            (Tile::Register(l), Tile::Register(r), Tile::Register(a)) => {
                register_execute(&l.data, &r.data, &mut a.data, a.config, a.product_type);
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
            (Tile::SharedMemory(shared), Tile::Mma(t)) => match ident {
                StageIdent::Lhs => mma_load_lhs_from_shared::<SE, SS, N, V, R, A, SIO>(
                    shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                    t.mma_io_config,
                ),
                StageIdent::Rhs => mma_load_rhs_from_shared::<SE, SS, N, V, L, A, SIO>(
                    shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                    t.mma_io_config,
                ),
                StageIdent::Acc => mma_load_acc_from_shared::<SE, SS, N, V, L, R, SIO>(
                    shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                    t.mma_io_config,
                ),
                _ => panic!("Invalid ident for copy_from (mma)"),
            },
            (Tile::None, Tile::Mma(t)) => {
                mma_load_acc_zeros::<SE, SS, N, V, L, R>(
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                    t.mma_io_config,
                );
            }

            // --- Register loads ---
            (Tile::SharedMemory(shared), Tile::Register(t)) => {
                register_load_from_shared::<SE, SS, N, V, SIO>(
                    shared,
                    &mut t.data,
                    t.matrix_layout,
                    t.config,
                    t.product_type,
                    ident,
                );
            }
            (Tile::None, Tile::Register(t)) => {
                register_load_zeros::<N, V>(&mut t.data, t.config, ident);
            }

            // --- PlaneVec loads ---
            (Tile::SharedMemory(shared), Tile::PlaneVec(t)) => {
                planevec_load_from_shared::<SE, SS, N, V, SIO>(
                    shared,
                    &mut t.data,
                    t.config,
                    ident,
                );
            }
            (Tile::None, Tile::PlaneVec(t)) => {
                planevec_load_zeros::<N, V>(&mut t.data, t.config);
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
            (Tile::Mma(t), Tile::SharedMemory(shared)) => {
                mma_write_to_shared::<N, V, SE, SS, L, R>(
                    shared,
                    &t.fragment,
                    t.config,
                    t.mma_io_config,
                );
            }
            (Tile::Register(t), Tile::SharedMemory(shared)) => {
                register_write_to_shared::<N, V, SE, SS>(shared, &t.data, t.config);
            }
            (Tile::PlaneVec(t), Tile::SharedMemory(shared)) => {
                planevec_write_to_shared::<N, V, SE, SS>(
                    shared,
                    &t.data,
                    t.config,
                    t.reduce_vector_size,
                );
            }
            (Tile::Interleaved(t), Tile::SharedMemory(shared)) => {
                interleaved_write_to_shared::<N, V, SE, SS>(shared, &t.data, t.config);
            }

            _ => panic!("Unsupported storage pair for copy_from"),
        }
    }
}
