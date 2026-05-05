use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{
        BounceTile, CmmaTile, InterleavedTile, MmaFragment, MmaFragmentExpand, MmaTile,
        PlaneVecTile, RegisterTile, ScopeMarker, SharedTile, TileScope, UnitTile, WhiteboxFragment,
        matmul::{
            cmma_execute, cmma_load_from_shared, cmma_load_zeros, cmma_write_to_shared,
            interleaved_execute, interleaved_load_from_shared, interleaved_load_zeros,
            interleaved_write_to_shared, mma_execute, mma_load_acc_from_shared, mma_load_acc_zeros,
            mma_load_lhs_from_shared, mma_load_rhs_from_shared, mma_write_to_shared,
            planevec_execute, planevec_load_from_shared, planevec_load_zeros,
            planevec_write_to_shared, register_execute, register_load_from_shared,
            register_load_zeros, register_write_to_shared,
        },
    },
};

#[derive(CubeType)]
pub enum Tile<N: Numeric, Sc: TileScope, IO: SliceVisibility> {
    SharedMemory(SharedTile<N, IO>),
    Cmma(CmmaTile<N>),
    Mma(MmaTile<N>),
    Register(RegisterTile<N>),
    PlaneVec(PlaneVecTile<N>),
    Interleaved(InterleavedTile<N>),
    /// Each unit holds a full row-major copy of the tile in registers.
    /// Only valid when `Sc = Unit`.
    Unit(UnitTile<N>),
    /// The tile is fragmented across plane units, with the layout exposed.
    /// Only valid when `Sc = Plane`.
    WhiteboxFragment(WhiteboxFragment<N>),
    /// Bundles a cmma fragment, an smem scratch slice, and a `WhiteboxFragment` view.
    /// From the caller's perspective it is a single tile; the smem round-trip
    /// is internal to ops dispatch. Only valid when `Sc = Plane`.
    Bounce(BounceTile<N>),
    Broadcasted(Value<N>),
    None,
    _Phantom(ScopeMarker<Sc>),
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
            (Tile::Mma(l), Tile::Mma(r), Tile::Mma(a)) => match &l.fragment {
                MmaFragment::Lhs(lf) => match &r.fragment {
                    MmaFragment::Rhs(rf) => match &mut a.fragment {
                        MmaFragment::Acc(af) => {
                            mma_execute(lf, rf, af, a.matrix_layout, a.config);
                        }
                        MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                            panic!("Mma: expected Acc role for accumulator")
                        }
                    },
                    MmaFragment::Lhs(_) | MmaFragment::Acc(_) => {
                        panic!("Mma: expected Rhs role for rhs")
                    }
                },
                MmaFragment::Rhs(_) | MmaFragment::Acc(_) => {
                    panic!("Mma: expected Lhs role for lhs")
                }
            },
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
            (Tile::SharedMemory(shared), Tile::Mma(t)) => {
                let shared = shared.view::<SS>();
                match &mut t.fragment {
                    MmaFragment::Lhs(f) => mma_load_lhs_from_shared::<SE, SS, N, R, A, SIO>(
                        &shared,
                        f,
                        t.matrix_layout,
                        t.config,
                    ),
                    MmaFragment::Rhs(f) => mma_load_rhs_from_shared::<SE, SS, N, L, A, SIO>(
                        &shared,
                        f,
                        t.matrix_layout,
                        t.config,
                    ),
                    MmaFragment::Acc(f) => mma_load_acc_from_shared::<SE, SS, N, L, R, SIO>(
                        &shared,
                        f,
                        t.matrix_layout,
                        t.config,
                    ),
                }
            }
            (Tile::None, Tile::Mma(t)) => match &mut t.fragment {
                MmaFragment::Acc(f) => {
                    mma_load_acc_zeros::<SE, SS, N, L, R>(f, t.matrix_layout, t.config);
                }
                MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                    panic!("Mma zero-load only supported for Acc role")
                }
            },

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
            (Tile::Mma(t), Tile::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                match &t.fragment {
                    MmaFragment::Acc(f) => {
                        mma_write_to_shared::<N, SS, SE, L, R>(&mut shared, f, t.config);
                    }
                    MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                        panic!("Mma write_to_shared only supported for Acc role")
                    }
                }
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
