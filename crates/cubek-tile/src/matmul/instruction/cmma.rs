//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`. The accumulator is always a
//! resident fragment; the operands arrive either as fragments (hand-wired) or as memory
//! windows (the schedules' staged path), staged into transient `A`/`B` fragments here.
//! [`mma_partition`] is the multi-fragment form: one instance's whole partition contracted
//! in comptime loops with `A`/`B` fragment reuse.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::matmul::resident::{contracted_axis, region_at};
use crate::*;

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// Tensor-core contraction `self += lhs · rhs`. Fragment operands execute directly;
    /// memory operands (what the lowering delivers: windows into a stage or gmem) are
    /// loaded into transient `A`/`B` fragments each call — only the accumulator is resident.
    pub(crate) fn mma<L: Numeric, R: Numeric>(&self, lhs: &Tile<L>, rhs: &Tile<R>) {
        match (&lhs.tile_kind, &rhs.tile_kind) {
            (TileKind::Cmma(a), TileKind::Cmma(b)) => {
                cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
            }
            (TileKind::Gmem(a) | TileKind::Smem(a), TileKind::Gmem(b) | TileKind::Smem(b)) => {
                // The tile is `m × k` on lhs and `k × n` on rhs (trailing two axes; any
                // leading batch axes are extent-1 at a final tile).
                let m = comptime!(lhs.space.extent_at(lhs.space.rank() - 2));
                let k = comptime!(lhs.space.extent_at(lhs.space.rank() - 1));
                let n = comptime!(rhs.space.extent_at(rhs.space.rank() - 1));

                // The rendezvous with the stage fill belongs to the schedule that filled it;
                // here the operands are simply readable memory.
                let mut a_frag = unsafe {
                    Matrix::<L>::uninitialized(MatrixIdent::A, m, n, k, MatrixLayout::RowMajor)
                };
                cmma::load(&mut a_frag, a.window_slice(), a.row_stride());
                let mut b_frag = unsafe {
                    Matrix::<R>::uninitialized(MatrixIdent::B, m, n, k, MatrixLayout::RowMajor)
                };
                cmma::load(&mut b_frag, b.window_slice(), b.row_stride());

                cmma::execute(&a_frag, &b_frag, &self.matrix, &self.matrix);
            }
            _ => panic!("cmma accumulator requires cmma or memory lhs and rhs"),
        }
    }
}

/// Contract an instance's whole fragment partition against its operand windows — the legacy
/// plane-partitioned loop nesting, fully comptime: per `ki`, `A` fragments are reused across
/// the `n` loop and each `B` fragment across the `m` loop.
#[cube]
pub(crate) fn mma_partition<Acc: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &mut Tile<Acc>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    // Fragment dims come from the final tile; the contraction walks `k_tiles` sub-tiles
    // of the contracted axis at this level.
    let fin = comptime!(acc.space.final_space());
    let m = comptime!(fin.extent_at(fin.rank() - 2));
    let n = comptime!(fin.extent_at(fin.rank() - 1));
    let rows = comptime!(acc.space.axis_at(acc.space.rank() - 2));
    let cols = comptime!(acc.space.axis_at(acc.space.rank() - 1));
    let contracted = comptime!(contracted_axis(&lhs.space, &acc.space));
    let k = comptime!(lhs.space.final_space().extent(contracted));
    let k_tiles = comptime!(lhs.space.count(contracted));

    match &acc.tile_kind {
        TileKind::CmmaPartition(part) => {
            let m_tiles = comptime!(part.m_tiles);
            let n_tiles = comptime!(part.n_tiles);
            // The contraction steps are a *runtime* loop: `ki` only positions memory
            // windows (fragment indices stay comptime), and unrolling it would keep every
            // step's transient fragments live at once.
            for ki in 0..k_tiles {
                let mut a_frags = Sequence::<CmmaData<Lhs>>::new();
                #[unroll]
                for mi in 0..m_tiles {
                    let mut a = CmmaData::<Lhs>::alloc(MatrixIdent::A, m, n, k);
                    let w = lhs.at(&region_at(
                        comptime!(lhs.space.clone()),
                        rows,
                        mi,
                        contracted,
                        ki,
                    ));
                    match &w.tile_kind {
                        TileKind::Gmem(s) | TileKind::Smem(s) => a.load_window(s),
                        TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                            panic!("mma_partition: lhs must be a memory window")
                        }
                    }
                    a_frags.push(a);
                }
                #[unroll]
                for ni in 0..n_tiles {
                    let mut b = CmmaData::<Rhs>::alloc(MatrixIdent::B, m, n, k);
                    let w = rhs.at(&region_at(
                        comptime!(rhs.space.clone()),
                        cols,
                        ni,
                        contracted,
                        ki,
                    ));
                    match &w.tile_kind {
                        TileKind::Gmem(s) | TileKind::Smem(s) => b.load_window(s),
                        TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                            panic!("mma_partition: rhs must be a memory window")
                        }
                    }
                    #[unroll]
                    for mi in 0..m_tiles {
                        let acc_frag = part.at(mi, ni);
                        cmma::execute(
                            &a_frags.index(mi).matrix,
                            &b.matrix,
                            &acc_frag.matrix,
                            &acc_frag.matrix,
                        );
                    }
                }
            }
        }
        TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
            panic!("mma_partition: accumulator is not a fragment partition")
        }
    }
}
