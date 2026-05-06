//! Per-flavor tile matmul compute: `*_execute`, `*_load_*`, `*_write_to_shared`,
//! plus the fragment readers/writers for each flavor. Tile data and matmul
//! configs live alongside the corresponding data structures in
//! [`crate::tile::data`].

pub mod cmma;
pub mod interleaved;
pub mod mma;
pub mod plane_vec;
pub mod register;

use cubecl::prelude::*;

use crate::tile::{
    MmaFragment, MmaFragmentExpand, Tile, TileExpand, TileScope,
    compute::matmul::{
        cmma::cmma_execute, interleaved::interleaved_execute, mma::mma_execute,
        plane_vec::planevec_execute, register::register_execute,
    },
};

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
}
