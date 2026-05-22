//! `Tile::mma` / `Tile::mma_partition` dispatchers and the partition
//! load/write-back helpers.

use cubecl::prelude::*;

use crate::{
    StageIdent,
    stage::Stage,
    tile::{
        PartitionScheduler, StageEventListener, Tile, TileExpand, TileKind, TileKindExpand,
        TileScope, WriteEvent, WriteEventListener,
    },
};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc> {
    /// `self += lhs · rhs`. For `(Stage, Stage, Partition)` use
    /// [`Tile::mma_partition`].
    pub fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &Tile<L, Sc>, rhs: &Tile<R, Sc>) {
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

    /// `mma` for `(Stage, Stage, Partition)` operands with rhs fragments
    /// held under `TileKind::Pipelined`.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_partition<
        LhsS: Numeric,
        LhsSize: Size,
        LhsR: Numeric,
        RhsS: Numeric,
        RhsSize: Size,
        RhsR: Numeric,
        SEL: StageEventListener,
    >(
        &mut self,
        lhs: &Tile<LhsS, Sc>,
        rhs: &Tile<RhsS, Sc>,
        a_fragment: &mut Sequence<Tile<LhsR, Sc>>,
        b_fragments: &mut Tile<RhsR, Sc>,
        #[comptime] partition_size_k: u32,
        listener: SEL,
        scheduler: &PartitionScheduler,
    ) {
        match (&lhs.kind, &rhs.kind, &mut self.kind, &mut b_fragments.kind) {
            (
                TileKind::Stage(a_stage),
                TileKind::Stage(b_stage),
                TileKind::Partition(acc),
                TileKind::Pipelined(b_frags),
            ) => acc.execute_with_listener::<LhsS, LhsSize, LhsR, RhsS, RhsSize, RhsR, SEL>(
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

#[cube]
/// Fill a partition accumulator from a stage. `None`-kind stage zero-inits.
pub fn load_partition_from_stage<
    AccSE: Numeric,
    AccSS: Size,
    LhsRE: Numeric,
    RhsRE: Numeric,
    AccRE: Numeric,
    Sc: TileScope,
    StageAcc: Stage<AccSE>,
>(
    stage: &StageAcc,
    acc: &mut Tile<AccRE, Sc>,
    scheduler: &PartitionScheduler,
    #[comptime] partition_size_m: u32,
    #[comptime] partition_size_n: u32,
) {
    let n_iterations = partition_size_n as usize;

    #[unroll]
    for m in 0..partition_size_m as usize {
        let m_stage = scheduler.map_m(m as u32);

        #[unroll]
        for n in 0..n_iterations {
            let n_stage = scheduler.map_n(n as u32);

            let acc_tile = acc.partition_tile_at_mut(m, n, n_iterations);
            let tile = StageAcc::tile::<Sc>(stage, (m_stage, n_stage));
            acc_tile.copy_from::<AccSE, AccSS, LhsRE, RhsRE, AccRE>(&tile, StageIdent::Acc);
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
/// Write a partition accumulator back to an output stage, emitting
/// `Begin` / `TileStored` / `Finish` events.
pub fn write_partition_to_stage<
    OutSE: Numeric,
    AccSS: Size,
    LhsRE: Numeric,
    RhsRE: Numeric,
    AccRE: Numeric,
    Sc: TileScope,
    OutStage: Stage<OutSE>,
    W: WriteEventListener,
>(
    acc: &mut Tile<AccRE, Sc>,
    out_stage: &mut OutStage,
    listener: &mut W,
    scheduler: &PartitionScheduler,
    #[comptime] partition_size_m: u32,
    #[comptime] partition_size_n: u32,
) {
    let n_iterations = partition_size_n as usize;

    W::on_event(listener, WriteEvent::new_Begin());

    #[unroll]
    for m_iter in 0..partition_size_m as usize {
        let m_store = scheduler.map_m(m_iter as u32);

        #[unroll]
        for n_iter in 0..n_iterations {
            let n_store = scheduler.map_n(n_iter as u32);

            let tile_accumulator = acc.partition_tile_at_mut(m_iter, n_iter, n_iterations);

            let tile_pos = (m_store, n_store);
            let mut tile = OutStage::tile::<Sc>(&*out_stage, tile_pos);

            tile.copy_from::<AccRE, AccSS, LhsRE, RhsRE, AccRE>(
                &*tile_accumulator,
                StageIdent::Out,
            );

            W::on_event(listener, WriteEvent::new_TileStored(tile_pos));
        }
    }

    W::on_event(listener, WriteEvent::new_Finish());
}
