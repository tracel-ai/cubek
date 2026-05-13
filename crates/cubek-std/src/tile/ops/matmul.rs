//! `Tile::mma` dispatcher. Each arm delegates to a `.mma()` method on the
//! variant's data struct in [`crate::tile::variants`] (where the matmul-execute,
//! load/write/zero-init helpers also live). Bounce arms route through their
//! inner CMMA fragment.

use cubecl::prelude::*;

use crate::tile::{Tile, TileExpand, TileKind, TileKindExpand, TileScope};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Executes `lhs · rhs`, accumulating the result into `self`.
    ///
    /// For instruction-level operands (Cmma/Mma/Register/PlaneVec/Interleaved/
    /// Bounce) the body delegates to the per-variant `.mma()` method. Stage-
    /// level inputs (`(Stage, Stage, Partition)`) need extra context the
    /// `.mma` signature can't carry (per-primitive instruction-tile fragments,
    /// the `PartitionScheduler`, an optional `StageEventListener`); that
    /// combination panics with a redirect to
    /// [`crate::tile::execute_partition_matmul`]. The arm is wired so the
    /// pattern surface is uniform and unsupported stage-level use produces a
    /// targeted error rather than the generic "Unsupported storage" fallback.
    pub fn mma<L: Numeric, LIO: SliceVisibility, R: Numeric, RIO: SliceVisibility>(
        &mut self,
        lhs: &Tile<L, Sc, LIO>,
        rhs: &Tile<R, Sc, RIO>,
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
            (TileKind::Stage(_), TileKind::Stage(_), TileKind::Partition(_)) => {
                panic!(
                    "Tile::mma: (Stage, Stage, Partition) requires extra context \
                     (per-primitive lhs/rhs fragments, PartitionScheduler, partition \
                     dimensions, optional StageEventListener) that .mma's signature \
                     can't carry — call execute_partition_matmul (or \
                     execute_partition_matmul_with_listener) directly."
                )
            }
            _ => panic!("Unsupported storage combination for mma"),
        }
    }
}
