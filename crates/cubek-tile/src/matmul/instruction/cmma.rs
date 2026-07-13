//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`. The accumulator is
//! always a resident fragment; the operands arrive as fragments or as staged smem
//! windows (row-major by construction), the latter loaded into transient `A`/`B`
//! fragments here. A gmem window's layout is unchecked, so it must be staged first.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// Tensor-core contraction `self += lhs · rhs`. Fragment operands execute directly;
    /// smem operands are loaded into transient `A`/`B` fragments each call. Smem only:
    /// the engine lays stages out row-major, which `cmma::load` assumes; a gmem
    /// window's layout is unchecked here, so it must be staged first.
    pub(crate) fn mma<L: Numeric, R: Numeric>(&self, lhs: &Tile<L>, rhs: &Tile<R>) {
        match (&lhs.tile_kind, &rhs.tile_kind) {
            (TileKind::Cmma(a), TileKind::Cmma(b)) => {
                cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
            }
            (TileKind::Smem(a), TileKind::Smem(b)) => {
                // The tile is `m × k` on lhs and `k × n` on rhs (trailing two axes; any
                // leading batch axes are extent-1 at a final tile).
                let m = comptime!(lhs.space.extent_at(lhs.space.rank() - 2));
                let k = comptime!(lhs.space.extent_at(lhs.space.rank() - 1));
                let n = comptime!(rhs.space.extent_at(rhs.space.rank() - 1));

                // The rendezvous with the stage fill belongs to the schedule that filled it.
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
            _ => panic!("cmma operands must be fragments or staged smem windows"),
        }
    }
}
