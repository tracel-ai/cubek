//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`. The accumulator is
//! always a resident fragment; the operands arrive as fragments or as staged smem
//! windows (row-major by construction), the latter loaded into transient `A`/`B`
//! fragments here. A gmem window's layout is unchecked, so it must be staged first.
//!
//! The rhs window is `{k, n}` as stored, or `{n, k}` when its trailing axis is the one
//! contracted: that window is read as a col-major `B`, so `a · bᵀ` needs no transposed
//! copy ([`rhs_layout`]).

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
            (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                (PlaneTile::Cmma(a), PlaneTile::Cmma(b)) => {
                    cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
                }
                _ => panic!("cmma operands must be cmma fragments"),
            },
            (TileKind::Smem(a), TileKind::Smem(b)) => {
                // The tile is `m × k` on lhs and `k × n` on rhs (trailing two axes; any
                // leading batch axes are extent-1 at a final tile), or `n × k` on a rhs
                // read col-major.
                let m = comptime!(lhs.space.extent_at(lhs.space.rank() - 2));
                let k = comptime!(lhs.space.extent_at(lhs.space.rank() - 1));
                let layout = comptime!(rhs_layout(
                    &rhs.space,
                    lhs.space.axis_at(lhs.space.rank() - 1)
                ));
                let n = comptime!(match layout {
                    MatrixLayout::RowMajor => rhs.space.extent_at(rhs.space.rank() - 1),
                    MatrixLayout::ColMajor => rhs.space.extent_at(rhs.space.rank() - 2),
                    MatrixLayout::Undefined => unreachable!(),
                });

                // The rendezvous with the stage fill belongs to the schedule that filled it.
                let mut a_frag = unsafe {
                    Matrix::<L>::uninitialized(MatrixIdent::A, m, n, k, MatrixLayout::RowMajor)
                };
                cmma::load(&mut a_frag, a.window_slice(), a.row_stride());
                let mut b_frag =
                    unsafe { Matrix::<R>::uninitialized(MatrixIdent::B, m, n, k, layout) };
                cmma::load(&mut b_frag, b.window_slice(), b.row_stride());

                cmma::execute(&a_frag, &b_frag, &self.matrix, &self.matrix);
            }
            _ => panic!("cmma operands must be fragments or staged smem windows"),
        }
    }
}

/// How a rhs window is read: col-major when its trailing axis is `contracted`, the matrix
/// then being the window's transpose; row-major otherwise. The rule
/// [`PlanePartition::store`](crate::PlanePartition::store) loads a `B` fragment by.
pub(crate) fn rhs_layout(rhs: &Space, contracted: Axis) -> MatrixLayout {
    match rhs.axis_at(rhs.rank() - 1) == contracted {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    }
}
