//! Partition-matmul body. Migrated from
//! `cubek_matmul::components::stage::matmul::partition::matmul`. Parameterized
//! over individual element types (no `MatmulTypes` dependency); the cubek-matmul
//! side plumbs the types through from its existing `MatmulTypes` extraction.
//!
//! The body takes concrete payload types (`StridedStage`, `PartitionTile`)
//! rather than wrapping `Tile<..>` everywhere — the `#[cube]` macro struggles
//! with reference-returning match arms over `TileKind`, so the
//! `(Stage, Stage, Partition)` `.mma` arm in `tile/ops/matmul.rs` is the place
//! that destructures the `TileKind` enum and forwards to this body.
//!
//! Note: only single-buffered rhs is supported in this body for now;
//! double-buffered support lands in a PR 4 follow-up — the cubecl `#[cube]`
//! macro hits trait-bound issues with the rhs-rotation pattern under generic
//! free-function context that don't appear in the cubek-matmul impl-method
//! context. The existing cubek-matmul `PartitionMatmul` continues to back
//! double-buffered flows until that is resolved.

use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{
        NoEvent, PartitionScheduler, PartitionTile, StageEvent, StageEventListener, StridedStage,
        Tile, TileScope,
        variants::stage::matmul::fragments::{RhsTile, RhsTileExpand},
    },
};

#[cube]
/// Execute the inner Tile Matmuls for one partition (single buffered rhs).
/// No event listener.
#[allow(clippy::too_many_arguments)]
pub fn execute_partition_matmul<
    LhsSE: Numeric,
    LhsSS: Size,
    LhsRE: Numeric,
    RhsSE: Numeric,
    RhsSS: Size,
    RhsRE: Numeric,
    AccRE: Numeric,
    Sc: TileScope,
>(
    lhs_stage: &StridedStage<LhsSE, ReadOnly>,
    rhs_stage: &StridedStage<RhsSE, ReadOnly>,
    lhs_fragment: &mut Sequence<Tile<LhsRE, Sc, ReadWrite>>,
    rhs_fragments: &mut RhsTile<Tile<RhsRE, Sc, ReadWrite>>,
    acc: &mut PartitionTile<AccRE, Sc, ReadWrite>,
    #[comptime] partition_size_m: u32,
    #[comptime] partition_size_n: u32,
    #[comptime] partition_size_k: u32,
    scheduler: &PartitionScheduler,
) {
    execute_partition_matmul_with_listener::<
        LhsSE,
        LhsSS,
        LhsRE,
        RhsSE,
        RhsSS,
        RhsRE,
        AccRE,
        Sc,
        NoEvent,
    >(
        lhs_stage,
        rhs_stage,
        lhs_fragment,
        rhs_fragments,
        acc,
        partition_size_m,
        partition_size_n,
        partition_size_k,
        NoEvent::new(),
        scheduler,
    );
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub fn execute_partition_matmul_with_listener<
    LhsSE: Numeric,
    LhsSS: Size,
    LhsRE: Numeric,
    RhsSE: Numeric,
    RhsSS: Size,
    RhsRE: Numeric,
    AccRE: Numeric,
    Sc: TileScope,
    SEL: StageEventListener,
>(
    lhs_stage: &StridedStage<LhsSE, ReadOnly>,
    rhs_stage: &StridedStage<RhsSE, ReadOnly>,
    lhs_fragment: &mut Sequence<Tile<LhsRE, Sc, ReadWrite>>,
    rhs_fragments: &mut RhsTile<Tile<RhsRE, Sc, ReadWrite>>,
    acc: &mut PartitionTile<AccRE, Sc, ReadWrite>,
    #[comptime] partition_size_m: u32,
    #[comptime] partition_size_n: u32,
    #[comptime] partition_size_k: u32,
    listener: SEL,
    scheduler: &PartitionScheduler,
) {
    match rhs_fragments {
        RhsTile::Single(rhs) => {
            execute_single::<LhsSE, LhsSS, LhsRE, RhsSE, RhsSS, RhsRE, AccRE, Sc, SEL>(
                lhs_stage,
                rhs_stage,
                lhs_fragment,
                rhs,
                acc,
                partition_size_m,
                partition_size_n,
                partition_size_k,
                listener,
                scheduler,
            )
        }
        RhsTile::Double(_rhs) => panic!(
            "execute_partition_matmul: Double buffering not yet supported in the cubek-std body \
             (PR 4 follow-up); existing cubek-matmul PartitionMatmul still backs double-buffered flows"
        ),
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn execute_single<
    LhsSE: Numeric,
    LhsSS: Size,
    LhsRE: Numeric,
    RhsSE: Numeric,
    RhsSS: Size,
    RhsRE: Numeric,
    AccRE: Numeric,
    Sc: TileScope,
    SEL: StageEventListener,
>(
    lhs_stage: &StridedStage<LhsSE, ReadOnly>,
    rhs_stage: &StridedStage<RhsSE, ReadOnly>,
    lhs_fragment: &mut Sequence<Tile<LhsRE, Sc, ReadWrite>>,
    rhs_fragment: &mut Tile<RhsRE, Sc, ReadWrite>,
    acc: &mut PartitionTile<AccRE, Sc, ReadWrite>,
    #[comptime] partition_size_m: u32,
    #[comptime] partition_size_n: u32,
    #[comptime] partition_size_k: u32,
    mut listener: SEL,
    scheduler: &PartitionScheduler,
) {
    SEL::on_event(&mut listener, StageEvent::Begin);

    let m_iterations = partition_size_m as usize;
    let n_iterations = partition_size_n as usize;
    let k_iterations = partition_size_k as usize;

    let mut lhs_load_counter = 0.comptime();
    let mut rhs_load_counter = 0.comptime();
    let mut execute_counter = 0.comptime();
    let lhs_load_total = (m_iterations * k_iterations) as u32;
    let rhs_load_total = (n_iterations * k_iterations) as u32;
    let execute_total = (m_iterations * n_iterations * k_iterations) as u32;

    #[unroll]
    for k_iter in 0..k_iterations {
        let k_load_iter = scheduler.map_k(k_iter as u32);

        #[unroll]
        for m_iter in 0..m_iterations {
            let m_load_iter = scheduler.map_m(m_iter as u32);

            let shared = lhs_stage.get_tile((m_load_iter, k_load_iter));
            let tile_lhs = Tile::new_SharedMemory(shared);

            lhs_fragment
                .index_mut(m_iter)
                .copy_from::<LhsSE, LhsSS, LhsRE, RhsRE, AccRE, ReadOnly>(
                    &tile_lhs,
                    StageIdent::Lhs,
                );

            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: lhs_load_counter,
                    total: lhs_load_total
                }],
            );
            comptime!(lhs_load_counter += 1);
        }

        #[unroll]
        for n_iter in 0..n_iterations {
            let n_load_iter = scheduler.map_n(n_iter as u32);

            let shared = rhs_stage.get_tile((k_load_iter, n_load_iter));
            let rhs_tile_next = Tile::new_SharedMemory(shared);

            rhs_fragment.copy_from::<RhsSE, RhsSS, LhsRE, RhsRE, AccRE, ReadOnly>(
                &rhs_tile_next,
                StageIdent::Rhs,
            );

            SEL::on_event(
                &mut listener,
                comptime![StageEvent::RhsLoaded {
                    current: rhs_load_counter,
                    total: rhs_load_total
                }],
            );
            comptime!(rhs_load_counter += 1);

            #[unroll]
            for m_iter in 0..m_iterations {
                let accumulator = acc.tiles.index_mut(m_iter * n_iterations + n_iter);
                accumulator.mma(&lhs_fragment[m_iter], rhs_fragment);

                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::TileMatmulCompleted {
                        current: execute_counter,
                        total: execute_total
                    }],
                );
                comptime!(execute_counter += 1);
            }
        }
    }

    assert!(lhs_load_counter == lhs_load_total);
    assert!(rhs_load_counter == rhs_load_total);
    assert!(execute_counter == execute_total);
    SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
}
