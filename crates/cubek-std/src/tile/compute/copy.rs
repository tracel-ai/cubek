use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{
        MmaFragment, MmaFragmentExpand, Tile, TileExpand, TileKind, TileKindExpand, TileScope,
        compute::matmul::{
            cmma::{cmma_load_from_shared, cmma_load_zeros, cmma_write_to_shared},
            interleaved::{
                interleaved_load_from_shared, interleaved_load_zeros, interleaved_write_to_shared,
            },
            mma::{
                mma_load_acc_from_shared, mma_load_acc_zeros, mma_load_lhs_from_shared,
                mma_load_rhs_from_shared, mma_write_to_shared,
            },
            plane_vec::{planevec_load_from_shared, planevec_load_zeros, planevec_write_to_shared},
            register::{register_load_from_shared, register_load_zeros, register_write_to_shared},
        },
    },
};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Zero-initializes the tile in place using the per-variant init that
    /// matches the storage's expected layout (e.g. `mma_load_acc_zeros` for
    /// MMA Acc, `cmma::fill` for CMMA fragments, plain register fills
    /// otherwise). This is the direct entry point for "I want a zeroed
    /// destination"; `copy_from` with a `Tile::None` source delegates here
    /// for the optional-stage flow.
    ///
    /// `L` / `R` are only consulted on the MMA path (which needs the matmul
    /// type triple at the layout-aware load); other variants ignore them.
    pub fn init_zero<L: Numeric, R: Numeric>(&mut self, #[comptime] ident: StageIdent) {
        match &mut self.kind {
            TileKind::Cmma(t) => cmma_load_zeros::<N>(&mut t.matrix),
            TileKind::Bounce(b) => cmma_load_zeros::<N>(&mut b.cmma.matrix),
            TileKind::Mma(t) => match &mut t.fragment {
                MmaFragment::Acc(f) => {
                    mma_load_acc_zeros::<N, L, R>(f, t.matrix_layout, t.config);
                }
                MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                    panic!("init_zero: Mma zero-init only supported for Acc role")
                }
            },
            TileKind::Register(t) => register_load_zeros::<N>(&mut t.data, t.config, ident),
            TileKind::PlaneVec(t) => planevec_load_zeros::<N>(&mut t.data, t.config),
            TileKind::Interleaved(t) => interleaved_load_zeros::<N>(&mut t.data, t.config),
            _ => panic!("init_zero: unsupported tile variant"),
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
        match (&source.kind, &mut self.kind) {
            // --- Zero-init from absent source ---
            // Mirrors `init_zero` (we can't call it here without releasing
            // the outer borrow; keeping the arms inline is the simplest
            // way to satisfy the borrow checker).
            (TileKind::None, TileKind::Cmma(t)) => {
                cmma_load_zeros::<N>(&mut t.matrix);
            }
            (TileKind::None, TileKind::Bounce(b)) => {
                cmma_load_zeros::<N>(&mut b.cmma.matrix);
            }
            (TileKind::None, TileKind::Mma(t)) => match &mut t.fragment {
                MmaFragment::Acc(f) => {
                    mma_load_acc_zeros::<N, L, R>(f, t.matrix_layout, t.config);
                }
                MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                    panic!("Mma zero-load only supported for Acc role")
                }
            },
            (TileKind::None, TileKind::Register(t)) => {
                register_load_zeros::<N>(&mut t.data, t.config, ident);
            }
            (TileKind::None, TileKind::PlaneVec(t)) => {
                planevec_load_zeros::<N>(&mut t.data, t.config);
            }
            (TileKind::None, TileKind::Interleaved(t)) => {
                interleaved_load_zeros::<N>(&mut t.data, t.config);
            }

            // --- Cmma loads ---
            (TileKind::SharedMemory(shared), TileKind::Cmma(t)) => {
                let shared = shared.view::<SS>();
                cmma_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut t.matrix,
                    ident,
                    t.matrix_layout,
                );
            }

            // --- Bounce loads (delegate to inner cmma) ---
            (TileKind::SharedMemory(shared), TileKind::Bounce(b)) => {
                let shared = shared.view::<SS>();
                cmma_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut b.cmma.matrix,
                    ident,
                    b.cmma.matrix_layout,
                );
            }

            // --- Mma loads ---
            (TileKind::SharedMemory(shared), TileKind::Mma(t)) => {
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

            // --- Register loads ---
            (TileKind::SharedMemory(shared), TileKind::Register(t)) => {
                let shared = shared.view::<SS>();
                register_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut t.data,
                    t.matrix_layout,
                    t.config,
                    ident,
                );
            }

            // --- PlaneVec loads ---
            (TileKind::SharedMemory(shared), TileKind::PlaneVec(t)) => {
                let shared = shared.view::<SS>();
                planevec_load_from_shared::<SE, SS, N, SIO>(&shared, &mut t.data, t.config, ident);
            }

            // --- Interleaved loads ---
            (TileKind::SharedMemory(shared), TileKind::Interleaved(t)) => {
                let shared = shared.view::<SS>();
                interleaved_load_from_shared::<SE, SS, N, SIO>(
                    &shared,
                    &mut t.data,
                    t.config,
                    ident,
                );
            }

            // --- Writes: shared memory copies from a compute container ---
            (TileKind::Cmma(t), TileKind::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                cmma_write_to_shared::<N, SS, SE>(&mut shared, &t.matrix);
            }
            (TileKind::Bounce(b), TileKind::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                cmma_write_to_shared::<N, SS, SE>(&mut shared, &b.cmma.matrix);
            }
            (TileKind::Mma(t), TileKind::SharedMemory(shared)) => {
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
            (TileKind::Register(t), TileKind::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                register_write_to_shared::<N, SS, SE>(&mut shared, &t.data, t.config);
            }
            (TileKind::PlaneVec(t), TileKind::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                planevec_write_to_shared::<SE, N, SS>(&mut shared, &t.data, t.config);
            }
            (TileKind::Interleaved(t), TileKind::SharedMemory(shared)) => {
                let mut shared = shared.view::<SS>();
                interleaved_write_to_shared::<N, SS, SE>(&mut shared, &t.data, t.config);
            }

            _ => panic!("Unsupported storage pair for copy_from"),
        }
    }
}
