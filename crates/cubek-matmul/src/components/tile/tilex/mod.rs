use cubecl::{cmma::Matrix, prelude::*};
use cubek_std::{
    MatrixLayout,
    tile::{StridedTile, mma::MmaIOConfig},
};

use crate::components::tile::{ProductType, SharedTileConfig};
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

// TODO
// pub struct Unit;
pub struct Plane;
// pub struct Cube;

// ===========================================================================
// Tilex: the main enum. Each variant holds its storage + comptime fields
// (matrix_layout, config, and any kind-specific data).
// ===========================================================================

#[derive(CubeType)]
pub enum Tilex<N: Numeric, V: Size, IO: SliceVisibility> {
    GlobalMemory(Slice<Vector<N, V>, IO>),
    SharedMemory(StridedTile<N, V, IO>),
    Cmma(CmmaTile<N>),
    Mma(MmaTile<N, V>),
    Register(RegisterTile<N>),
    PlaneVec(PlaneVecTile<N, V>),
    Interleaved(InterleavedTile<N>),
    Broadcasted(Value<N>),
    None,
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

// ===========================================================================
// Allocate functions are per-kind only (cmma_allocate_lhs, register_allocate_lhs, etc.)
// No dispatch needed — the caller (each TileMatmul impl) knows which kind it wants.
// ===========================================================================

// ===========================================================================
// Dispatch: execute — single match, extracts inner storage, calls impl fn.
// ===========================================================================

#[cube]
pub fn tilex_execute<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>(
    lhs: &Tilex<L, VL, ReadWrite>,
    rhs: &Tilex<R, VR, ReadWrite>,
    acc: &mut Tilex<A, VA, ReadWrite>,
) {
    match (lhs, rhs, acc) {
        (Tilex::Cmma(l), Tilex::Cmma(r), Tilex::Cmma(a)) => {
            cmma_execute(&l.matrix, &r.matrix, &mut a.matrix);
        }
        (Tilex::Mma(l), Tilex::Mma(r), Tilex::Mma(a)) => {
            mma_execute(
                &l.fragment,
                &r.fragment,
                &mut a.fragment,
                a.matrix_layout,
                a.config,
                a.mma_io_config,
            );
        }
        (Tilex::Register(l), Tilex::Register(r), Tilex::Register(a)) => {
            register_execute(&l.data, &r.data, &mut a.data, a.config, a.product_type);
        }
        (Tilex::PlaneVec(l), Tilex::PlaneVec(r), Tilex::PlaneVec(a)) => {
            planevec_execute(&l.data, &r.data, &mut a.data, a.config);
        }
        (Tilex::Interleaved(l), Tilex::Interleaved(r), Tilex::Interleaved(a)) => {
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
        _ => panic!("Unsupported storage combination for tilex_execute"),
    }
}

// ===========================================================================
// Dispatch: load — single match on (source, dest) pair.
// ===========================================================================

#[cube]
pub fn tilex_load<
    SE: Numeric,
    SS: Size,
    DE: Numeric,
    DS: Size,
    L: Numeric,
    R: Numeric,
    A: Numeric,
>(
    source: &Tilex<SE, SS, ReadOnly>,
    dest: &mut Tilex<DE, DS, ReadWrite>,
    #[comptime] ident: StageIdent,
) {
    match (source, dest) {
        // --- Cmma loads ---
        (Tilex::SharedMemory(shared), Tilex::Cmma(t)) => {
            cmma_load_from_shared::<SE, SS, DE, DS>(shared, &mut t.matrix, ident, t.matrix_layout);
        }
        (Tilex::None, Tilex::Cmma(t)) => {
            cmma_load_zeros::<DE, DS>(&mut t.matrix);
        }

        // --- Mma loads ---
        (Tilex::SharedMemory(shared), Tilex::Mma(t)) => match ident {
            StageIdent::Lhs => mma_load_lhs_from_shared::<SE, SS, DE, DS, R, A>(
                shared,
                &mut t.fragment,
                t.matrix_layout,
                t.config,
                t.mma_io_config,
            ),
            StageIdent::Rhs => mma_load_rhs_from_shared::<SE, SS, DE, DS, L, A>(
                shared,
                &mut t.fragment,
                t.matrix_layout,
                t.config,
                t.mma_io_config,
            ),
            StageIdent::Acc => mma_load_acc_from_shared::<SE, SS, DE, DS, L, R>(
                shared,
                &mut t.fragment,
                t.matrix_layout,
                t.config,
                t.mma_io_config,
            ),
            _ => panic!("Invalid ident for mma_load"),
        },
        (Tilex::None, Tilex::Mma(t)) => {
            mma_load_acc_zeros::<SE, SS, DE, DS, L, R>(
                &mut t.fragment,
                t.matrix_layout,
                t.config,
                t.mma_io_config,
            );
        }

        // --- Register loads ---
        (Tilex::SharedMemory(shared), Tilex::Register(t)) => {
            register_load_from_shared::<SE, SS, DE, DS>(
                shared,
                &mut t.data,
                t.matrix_layout,
                t.config,
                t.product_type,
                ident,
            );
        }
        (Tilex::None, Tilex::Register(t)) => {
            register_load_zeros::<DE, DS>(&mut t.data, t.config, ident);
        }

        // --- PlaneVec loads ---
        (Tilex::SharedMemory(shared), Tilex::PlaneVec(t)) => {
            planevec_load_from_shared::<SE, SS, DE, DS>(shared, &mut t.data, t.config, ident);
        }
        (Tilex::None, Tilex::PlaneVec(t)) => {
            planevec_load_zeros::<DE, DS>(&mut t.data, t.config);
        }

        // --- Interleaved loads ---
        (Tilex::SharedMemory(shared), Tilex::Interleaved(t)) => {
            interleaved_load_from_shared::<SE, SS, DE, DS>(shared, &mut t.data, t.config, ident);
        }
        (Tilex::None, Tilex::Interleaved(t)) => {
            interleaved_load_zeros::<DE, DS>(&mut t.data, t.config);
        }

        _ => panic!("Unsupported storage pair for tilex_load"),
    }
}

// ===========================================================================
// Dispatch: write — single match on (dest_stage, acc) pair.
// ===========================================================================

#[cube]
pub fn tilex_write<E: Numeric, ES: Size, A: Numeric, VA: Size, L: Numeric, R: Numeric>(
    tile: &mut Tilex<E, ES, ReadWrite>,
    out: &mut Tilex<A, VA, ReadWrite>,
) {
    match (tile, out) {
        (Tilex::SharedMemory(shared), Tilex::Cmma(t)) => {
            cmma_write_to_shared::<E, ES, A, VA>(shared, &t.matrix);
        }
        (Tilex::SharedMemory(shared), Tilex::Mma(t)) => {
            mma_write_to_shared::<E, ES, A, VA, L, R>(
                shared,
                &t.fragment,
                t.config,
                t.mma_io_config,
            );
        }
        (Tilex::SharedMemory(shared), Tilex::Register(t)) => {
            register_write_to_shared::<E, ES, A, VA>(shared, &t.data, t.config);
        }
        (Tilex::SharedMemory(shared), Tilex::PlaneVec(t)) => {
            planevec_write_to_shared::<E, ES, A, VA>(
                shared,
                &mut t.data,
                t.config,
                t.reduce_vector_size,
            );
        }
        (Tilex::SharedMemory(shared), Tilex::Interleaved(t)) => {
            interleaved_write_to_shared::<E, ES, A, VA>(shared, &mut t.data, t.config);
        }
        _ => panic!("Unsupported storage pair for tilex_write"),
    }
}
