//! `Tile::mma` dispatcher. Each arm delegates to a `.mma()` method on the
//! variant's data struct in [`crate::tile::variants`] (where the matmul-execute,
//! load/write/zero-init helpers also live). Bounce arms route through their
//! inner CMMA fragment.

use cubecl::prelude::*;

use crate::tile::{Tile, TileExpand, TileKind, TileKindExpand, TileScope};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Executes `lhs · rhs`, accumulating the result into `self`.
    pub fn mma<L: Numeric, R: Numeric>(
        &mut self,
        lhs: &Tile<L, Sc, ReadWrite>,
        rhs: &Tile<R, Sc, ReadWrite>,
    ) {
        match (&lhs.kind, &rhs.kind, &mut self.kind) {
            (TileKind::Cmma(l), TileKind::Cmma(r), TileKind::Cmma(a)) => a.mma(l, r),
            (TileKind::Cmma(l), TileKind::Cmma(r), TileKind::Bounce(a)) => a.cmma.mma(l, r),
            (TileKind::Bounce(l), TileKind::Cmma(r), TileKind::Bounce(a)) => a.cmma.mma(&l.cmma, r),
            (TileKind::Bounce(l), TileKind::Cmma(r), TileKind::Cmma(a)) => a.mma(&l.cmma, r),
            (TileKind::Mma(l), TileKind::Mma(r), TileKind::Mma(a)) => a.mma(l, r),
            (TileKind::Register(l), TileKind::Register(r), TileKind::Register(a)) => a.mma(l, r),
            (TileKind::PlaneVec(l), TileKind::PlaneVec(r), TileKind::PlaneVec(a)) => a.mma(l, r),
            (TileKind::Interleaved(l), TileKind::Interleaved(r), TileKind::Interleaved(a)) => {
                a.mma(l, r)
            }
            _ => panic!("Unsupported storage combination for mma"),
        }
    }
}
