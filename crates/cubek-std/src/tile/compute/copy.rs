use cubecl;
use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{
        MmaFragment, MmaFragmentExpand, Tile, TileExpand, TileScope,
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
        data::BounceTile,
    },
};

/// Internal `copy_from` between the `cmma` and `fragment` parts of a
/// [`BounceTile`]: cmma -> smem -> fragment. Used by the high-level
/// `softmax` / `scale_mul` / `scale_div` methods to make the fragment view
/// current.
#[cube]
pub(crate) fn cmma_to_whitebox_fragment<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    cubecl::cmma::store(
        &mut b.smem,
        &b.cmma.matrix,
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    b.fragment.load_from_slice(&b.smem.to_slice());
    sync_cube();
}

/// Internal `copy_from` between the `fragment` and `cmma` parts of a
/// [`BounceTile`]: fragment -> smem -> cmma. Reverses
/// [`cmma_to_whitebox_fragment`].
#[cube]
pub(crate) fn whitebox_fragment_to_cmma<E: Float>(b: &mut BounceTile<E>) {
    let stride = comptime!(b.cmma.tile_size.n());
    b.fragment.store_to(&mut b.smem);
    sync_cube();
    cubecl::cmma::load_with_layout(
        &b.cmma.matrix,
        &b.smem.to_slice(),
        stride,
        cubecl::cmma::MatrixLayout::RowMajor,
    );
}

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
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
