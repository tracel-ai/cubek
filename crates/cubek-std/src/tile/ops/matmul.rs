//! `Tile::mma` dispatcher. Each arm delegates to a `.mma()` method on the
//! variant's data struct in [`crate::tile::variants`]. Bounce arms route
//! through their inner CMMA fragment.
//!
//! The `(Stage, Stage, Partition)` combination needs extra context that
//! `.mma`'s three-tile signature can't carry. [`Tile::mma_partition`] is the
//! .mma-family entry point for it: it destructures the kinds and forwards to
//! [`PartitionTile::execute_with_listener`](crate::tile::PartitionTile).
//! Single- vs double-buffered rhs is encoded in `b_fragments`'s tile kind
//! ([`TileKind::Pipelined`] payload: a `Sequence` of length 1 or 2).

use cubecl::prelude::*;

use crate::tile::{
    PartitionScheduler, StageEventListener, Tile, TileExpand, TileKind, TileKindExpand, TileScope,
};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Executes `lhs · rhs`, accumulating the result into `self`. For
    /// instruction-level operands the body delegates to the per-variant
    /// `.mma()` method. The `(Stage, Stage, Partition)` arm panics — use
    /// [`Tile::mma_partition`] for that case.
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
                    "Tile::mma: (Stage, Stage, Partition) requires extra context — call \
                     Tile::mma_partition."
                )
            }
            _ => panic!("Unsupported storage combination for mma"),
        }
    }

    /// `.mma`-family entry point for `(Stage, Stage, Partition)` operands
    /// with rhs fragments held under [`TileKind::Pipelined`].
    /// Destructures the `Stage` payloads of `lhs` / `rhs`, the
    /// `Partition` payload of `self`, and the `Pipelined` payload of
    /// `b_fragments`, then forwards to
    /// [`PartitionTile::execute_with_listener`](crate::tile::PartitionTile).
    /// Panics if any operand is not the expected kind.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_partition<
        ASE: Numeric,
        ASS: Size,
        ARE: Numeric,
        BSE: Numeric,
        BSS: Size,
        BRE: Numeric,
        SEL: StageEventListener,
    >(
        &mut self,
        lhs: &Tile<ASE, Sc, ReadOnly>,
        rhs: &Tile<BSE, Sc, ReadOnly>,
        a_fragment: &mut Sequence<Tile<ARE, Sc, ReadWrite>>,
        b_fragments: &mut Tile<BRE, Sc, ReadWrite>,
        #[comptime] partition_size_k: u32,
        listener: SEL,
        scheduler: &PartitionScheduler,
    ) {
        match (
            &lhs.kind,
            &rhs.kind,
            &mut self.kind,
            &mut b_fragments.kind,
        ) {
            (
                TileKind::Stage(a_stage),
                TileKind::Stage(b_stage),
                TileKind::Partition(acc),
                TileKind::Pipelined(b_frags),
            ) => acc.execute_with_listener::<ASE, ASS, ARE, BSE, BSS, BRE, SEL>(
                a_stage,
                b_stage,
                a_fragment,
                b_frags,
                partition_size_k,
                listener,
                scheduler,
            ),
            _ => panic!(
                "Tile::mma_partition: requires (lhs, rhs, self, b_fragments) kinds = \
                 (Stage, Stage, Partition, Pipelined)"
            ),
        }
    }
}
