pub mod ops;
pub mod scope;
pub mod variants;

mod strided_tile;
mod tile_kind;

pub use ops::*;
pub use scope::{Cube, Plane, Scope, ScopeKind, ScopeMarker, Unit};
pub use strided_tile::*;
pub use tile_kind::*;
pub use variants::bounce_tile::*;
pub use variants::cmma::*;
pub use variants::interleaved::*;
pub use variants::local_tile::*;
pub use variants::mma::*;
pub use variants::plane_vec_mat_inner_product::*;
pub use variants::register::*;
pub use variants::unit_tile::*;

// Re-export the variant modules at their old paths so existing consumers keep
// working (e.g. `cubek_std::tile::cmma`, `cubek_std::tile::mma`).
pub use variants::{cmma, interleaved, mma, plane_vec_mat_inner_product, register};

use cubecl::cmma::Matrix as CubeMatrix;
use cubecl::prelude::*;

use crate::{MatrixLayout, StageIdent, tile::scope::Scope as TileScope};

#[derive(CubeType)]
pub enum Tile<N: Numeric, Sc: TileScope, IO: SliceVisibility> {
    SharedMemory(SharedTile<N, IO>),
    Cmma(CmmaTile<N>),
    MmaLhs(MmaLhsTile<N>),
    MmaRhs(MmaRhsTile<N>),
    MmaAcc(MmaAccTile<N>),
    Register(RegisterTile<N>),
    PlaneVec(PlaneVecTile<N>),
    Interleaved(InterleavedTile<N>),
    /// Each unit holds a full row-major copy of the tile in registers.
    /// Only valid when `Sc = Unit`.
    Unit(UnitTile<N>),
    /// The tile is fragmented across plane units. Only valid when `Sc = Plane`.
    Local(LocalTile<N>),
    /// Bundles a cmma fragment, an smem scratch slice, and a `LocalTile` view.
    /// From the caller's perspective it is a single tile; the smem round-trip
    /// is internal to ops dispatch. Only valid when `Sc = Plane`.
    Bounce(BounceTile<N>),
    Broadcasted(Value<N>),
    None,
    _Phantom(ScopeMarker<Sc>),
}

#[derive(CubeType)]
pub struct CmmaTile<N: Numeric> {
    pub matrix: CubeMatrix<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub tile_size: crate::TileSize,
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
pub struct PlaneVecTile<N: Numeric> {
    // Fragment inner size is `NPlaneVec` (= reduce_vector_size).
    pub data: Array<Vector<N, NPlaneVec>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: PlaneVecMatInnerProduct,
}

/// Wrapper over val to make enum work
#[derive(CubeType)]
pub struct Value<E: Numeric> {
    pub val: E,
}

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Executes `lhs · rhs`, accumulating the result into `self`.
    pub fn mma<L: Numeric, R: Numeric>(
        &mut self,
        lhs: &Tile<L, Sc, ReadWrite>,
        rhs: &Tile<R, Sc, ReadWrite>,
    ) {
        match (lhs, rhs, self) {
            (Tile::Cmma(l), Tile::Cmma(r), Tile::Cmma(a)) => {
                cmma_execute(&l.matrix, &r.matrix, &mut a.matrix);
            }
            (Tile::Cmma(l), Tile::Cmma(r), Tile::Bounce(a)) => {
                cmma_execute(&l.matrix, &r.matrix, &mut a.cmma.matrix);
            }
            (Tile::Bounce(l), Tile::Cmma(r), Tile::Bounce(a)) => {
                cmma_execute(&l.cmma.matrix, &r.matrix, &mut a.cmma.matrix);
            }
            (Tile::Bounce(l), Tile::Cmma(r), Tile::Cmma(a)) => {
                cmma_execute(&l.cmma.matrix, &r.matrix, &mut a.matrix);
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
    ///
    /// `SS` is the vector size of the shared memory tile involved in the copy
    /// (whether that's the source on a load, or the destination on a write).
    /// `L`/`R`/`A` are the matrix-level numeric types needed by the MMA
    /// readers/writers — they are unused on non-MMA paths.
    pub fn copy_from<
        SE: Numeric,
        SS: Size,
        L: Numeric,
        R: Numeric,
        A: Numeric,
        SIO: SliceVisibility,
    >(
        &mut self,
        source: &Tile<SE, Sc, SIO>,
        #[comptime] ident: StageIdent,
    ) {
        match (source, self) {
            // --- Cmma loads ---
            (Tile::SharedMemory(shared), Tile::Cmma(t)) => {
                let shared = shared.view::<SS>();
                cmma_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut t.matrix,
                    ident,
                    t.matrix_layout,
                );
            }
            (Tile::None, Tile::Cmma(t)) => {
                cmma_load_zeros::<N>(&mut t.matrix);
            }

            // --- Bounce loads (delegate to inner cmma) ---
            (Tile::SharedMemory(shared), Tile::Bounce(b)) => {
                let shared = shared.view::<SS>();
                cmma_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut b.cmma.matrix,
                    ident,
                    b.cmma.matrix_layout,
                );
            }
            (Tile::None, Tile::Bounce(b)) => {
                cmma_load_zeros::<N>(&mut b.cmma.matrix);
            }

            // --- Mma loads ---
            (Tile::SharedMemory(shared), Tile::MmaLhs(t)) => {
                let shared = shared.view::<SS>();
                mma_load_lhs_from_shared::<SE, SS, N, R, A, SIO>(
                    &shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                );
            }
            (Tile::SharedMemory(shared), Tile::MmaRhs(t)) => {
                let shared = shared.view::<SS>();
                mma_load_rhs_from_shared::<SE, SS, N, L, A, SIO>(
                    &shared,
                    &mut t.fragment,
                    t.matrix_layout,
                    t.config,
                );
            }
            (Tile::SharedMemory(shared), Tile::MmaAcc(t)) => {
                let shared = shared.view::<SS>();
                mma_load_acc_from_shared::<SE, SS, N, L, R, SIO>(
                    &shared,
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
                let shared = shared.view::<SS>();
                register_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut t.data,
                    t.matrix_layout,
                    t.config,
                    ident,
                );
            }
            (Tile::None, Tile::Register(t)) => {
                register_load_zeros::<N>(&mut t.data, t.config, ident);
            }

            // --- PlaneVec loads ---
            (Tile::SharedMemory(shared), Tile::PlaneVec(t)) => {
                let shared = shared.view::<SS>();
                planevec_load_from_shared::<SE, SS, N, SIO>(&shared, &mut t.data, t.config, ident);
            }
            (Tile::None, Tile::PlaneVec(t)) => {
                planevec_load_zeros::<N>(&mut t.data, t.config);
            }

            // --- Interleaved loads ---
            (Tile::SharedMemory(shared), Tile::Interleaved(t)) => {
                let shared = shared.view::<SS>();
                interleaved_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut t.data,
                    t.config,
                    ident,
                );
            }
            (Tile::None, Tile::Interleaved(t)) => {
                interleaved_load_zeros::<N>(&mut t.data, t.config);
            }

            // --- Writes: shared memory copies from a compute container ---
            (Tile::Cmma(t), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                cmma_write_to_shared::<N, SS, SE>(&mut shared, &t.matrix);
            }
            (Tile::Bounce(b), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                cmma_write_to_shared::<N, SS, SE>(&mut shared, &b.cmma.matrix);
            }
            (Tile::MmaAcc(t), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                mma_write_to_shared::<N, SS, SE, L, R>(&mut shared, &t.fragment, t.config);
            }
            (Tile::Register(t), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                register_write_to_shared::<N, SS, SE>(&mut shared, &t.data, t.config);
            }
            (Tile::PlaneVec(t), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                planevec_write_to_shared::<SE, N, SS>(&mut shared, &t.data, t.config);
            }
            (Tile::Interleaved(t), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                interleaved_write_to_shared::<N, SS, SE>(&mut shared, &t.data, t.config);
            }

            _ => panic!("Unsupported storage pair for copy_from"),
        }
    }
}
