//! `PartitionTile` (per-partition accumulator tiles) and `PipelinedTile`
//! (1 or 2 rhs register fragments) plus the partition matmul body.

use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{PartitionScheduler, StageEvent, StageEventListener, StageTile, Tile, TileScope},
};

/// Per-partition collection of instruction-level tiles, flattened in
/// `mn`-major order.
#[derive(CubeType)]
pub struct PartitionTile<N: Numeric, Sc: TileScope> {
    pub tiles: Sequence<Tile<N, Sc>>,
    #[cube(comptime)]
    pub rows: u32,
    #[cube(comptime)]
    pub cols: u32,
    #[cube(comptime)]
    pub _phantom: PhantomData<Sc>,
}

/// Rhs register fragments for the partition matmul. `fragments` has comptime
/// length 1 (single-buffered) or 2 (double-buffered with rotation).
#[derive(CubeType)]
pub struct PipelinedTile<N: Numeric, Sc: TileScope> {
    pub fragments: Sequence<Tile<N, Sc>>,
}

#[cube]
pub(crate) fn partition_get_at_mut<E: Numeric, Sc: TileScope>(
    partition: &mut PartitionTile<E, Sc>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] n_cols: usize,
) -> &mut Tile<E, Sc> {
    partition.tiles.index_mut(m * n_cols + n)
}

#[cube]
impl<CRE: Numeric, Sc: TileScope> PartitionTile<CRE, Sc> {
    /// Run the partition matmul. `b_fragments` length picks single (1) or
    /// double (2) buffering.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_listener<
        LhsS: Numeric,
        LhsSize: Size,
        LhsR: Numeric,
        RhsS: Numeric,
        RhsSize: Size,
        RhsR: Numeric,
        SEL: StageEventListener,
    >(
        &mut self,
        a_stage: &StageTile<LhsS>,
        b_stage: &StageTile<RhsS>,
        a_fragment: &mut Sequence<Tile<LhsR, Sc>>,
        b_fragments: &mut PipelinedTile<RhsR, Sc>,
        #[comptime] partition_size_k: u32,
        listener: SEL,
        scheduler: &PartitionScheduler,
    ) {
        let n_buffers = comptime!(b_fragments.fragments.len());
        if n_buffers == 1 {
            execute_single::<LhsS, LhsSize, LhsR, RhsS, RhsSize, RhsR, CRE, Sc, SEL>(
                a_stage,
                b_stage,
                a_fragment,
                &mut b_fragments.fragments,
                self,
                partition_size_k,
                listener,
                scheduler,
            );
        } else if n_buffers == 2 {
            execute_double::<LhsS, LhsSize, LhsR, RhsS, RhsSize, RhsR, CRE, Sc, SEL>(
                a_stage,
                b_stage,
                a_fragment,
                &mut b_fragments.fragments,
                self,
                partition_size_k,
                listener,
                scheduler,
            );
        } else {
            panic!("PartitionTile::execute_with_listener: b_fragments must have length 1 or 2");
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn execute_single<
    LhsS: Numeric,
    LhsSize: Size,
    LhsR: Numeric,
    RhsS: Numeric,
    RhsSize: Size,
    RhsR: Numeric,
    Acc: Numeric,
    Sc: TileScope,
    SEL: StageEventListener,
>(
    a_stage: &StageTile<LhsS>,
    b_stage: &StageTile<RhsS>,
    a_fragment: &mut Sequence<Tile<LhsR, Sc>>,
    b_fragments: &mut Sequence<Tile<RhsR, Sc>>,
    acc: &mut PartitionTile<Acc, Sc>,
    #[comptime] partition_size_k: u32,
    mut listener: SEL,
    scheduler: &PartitionScheduler,
) {
    SEL::on_event(&mut listener, StageEvent::Begin);

    let m_iterations = comptime!(acc.rows) as usize;
    let n_iterations = comptime!(acc.cols) as usize;
    let k_iterations = partition_size_k as usize;

    let mut a_load_counter = 0u32.comptime();
    let mut b_load_counter = 0u32.comptime();
    let mut execute_counter = 0u32.comptime();
    let a_load_total = (m_iterations * k_iterations) as u32;
    let b_load_total = (n_iterations * k_iterations) as u32;
    let execute_total = (m_iterations * n_iterations * k_iterations) as u32;

    #[unroll]
    for k_iter in 0..k_iterations {
        let k_load_iter = scheduler.map_k(k_iter as u32);

        #[unroll]
        for m_iter in 0..m_iterations {
            let m_load_iter = scheduler.map_m(m_iter as u32);

            let shared = a_stage.get_tile((m_load_iter, k_load_iter));
            let tile_lhs = Tile::new_SharedTile(shared);

            a_fragment
                .index_mut(m_iter)
                .copy_from::<LhsS, LhsSize, LhsR, RhsR, Acc>(&tile_lhs, StageIdent::Lhs);

            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: a_load_counter,
                    total: a_load_total
                }],
            );
            comptime!(a_load_counter += 1);
        }

        #[unroll]
        for n_iter in 0..n_iterations {
            let n_load_iter = scheduler.map_n(n_iter as u32);

            let shared = b_stage.get_tile((k_load_iter, n_load_iter));
            let rhs_tile_next = Tile::new_SharedTile(shared);

            b_fragments
                .index_mut(0usize)
                .copy_from::<RhsS, RhsSize, LhsR, RhsR, Acc>(&rhs_tile_next, StageIdent::Rhs);

            SEL::on_event(
                &mut listener,
                comptime![StageEvent::RhsLoaded {
                    current: b_load_counter,
                    total: b_load_total
                }],
            );
            comptime!(b_load_counter += 1);

            #[unroll]
            for m_iter in 0..m_iterations {
                let accumulator = acc.tiles.index_mut(m_iter * n_iterations + n_iter);
                accumulator.mma(&a_fragment[m_iter], b_fragments.index(0usize));

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

    assert!(a_load_counter == a_load_total);
    assert!(b_load_counter == b_load_total);
    assert!(execute_counter == execute_total);
    SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn execute_double<
    LhsS: Numeric,
    LhsSize: Size,
    LhsR: Numeric,
    RhsS: Numeric,
    RhsSize: Size,
    RhsR: Numeric,
    Acc: Numeric,
    Sc: TileScope,
    SEL: StageEventListener,
>(
    a_stage: &StageTile<LhsS>,
    b_stage: &StageTile<RhsS>,
    a_fragment: &mut Sequence<Tile<LhsR, Sc>>,
    b_fragments: &mut Sequence<Tile<RhsR, Sc>>,
    acc: &mut PartitionTile<Acc, Sc>,
    #[comptime] partition_size_k: u32,
    mut listener: SEL,
    scheduler: &PartitionScheduler,
) {
    SEL::on_event(&mut listener, StageEvent::Begin);

    let m_iterations = comptime!(acc.rows) as usize;
    let n_iterations = comptime!(acc.cols) as usize;
    let k_iterations = partition_size_k as usize;

    let mut a_load_counter = 0u32.comptime();
    let mut b_load_counter = 0u32.comptime();
    let mut execute_counter = 0u32.comptime();
    let a_load_total = (m_iterations * k_iterations) as u32;
    let b_load_total = (n_iterations * k_iterations) as u32;
    let execute_total = (m_iterations * n_iterations * k_iterations) as u32;

    #[unroll]
    for k_iter in 0..k_iterations {
        let k_load_iter = scheduler.map_k(k_iter as u32);

        #[unroll]
        for m_iter in 0..m_iterations {
            let m_load_iter = scheduler.map_m(m_iter as u32);

            let shared = a_stage.get_tile((m_load_iter, k_load_iter));
            let tile_lhs = Tile::new_SharedTile(shared);

            a_fragment
                .index_mut(m_iter)
                .copy_from::<LhsS, LhsSize, LhsR, RhsR, Acc>(&tile_lhs, StageIdent::Lhs);

            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: a_load_counter,
                    total: a_load_total
                }],
            );
            comptime!(a_load_counter += 1);
        }

        // Pre-load rhs[0] into slot 0.
        let first_load_iter = scheduler.map_n(0u32);
        let shared_first = b_stage.get_tile((k_load_iter, first_load_iter));
        let rhs_tile_first = Tile::new_SharedTile(shared_first);
        b_fragments
            .index_mut(0usize)
            .copy_from::<RhsS, RhsSize, LhsR, RhsR, Acc>(&rhs_tile_first, StageIdent::Rhs);

        SEL::on_event(
            &mut listener,
            comptime!(StageEvent::RhsLoaded {
                current: b_load_counter,
                total: b_load_total
            }),
        );
        comptime!(b_load_counter += 1);

        #[unroll]
        for n_iter in 1..n_iterations {
            let current_idx = (n_iter - 1) % 2;
            let next_idx = n_iter % 2;

            // Load rhs[n_iter] into the opposite slot.
            let n_load_iter = scheduler.map_n(n_iter as u32);
            let shared = b_stage.get_tile((k_load_iter, n_load_iter));
            let rhs_tile_next = Tile::new_SharedTile(shared);
            b_fragments
                .index_mut(next_idx)
                .copy_from::<RhsS, RhsSize, LhsR, RhsR, Acc>(&rhs_tile_next, StageIdent::Rhs);

            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: b_load_counter,
                    total: b_load_total
                }),
            );
            comptime!(b_load_counter += 1);

            // mma using rhs[n_iter - 1] from the current slot.
            let prev_n = n_iter - 1;
            #[unroll]
            for m_iter in 0..m_iterations {
                let accumulator = acc.tiles.index_mut(m_iter * n_iterations + prev_n);
                accumulator.mma(&a_fragment[m_iter], b_fragments.index(current_idx));

                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::TileMatmulCompleted {
                        current: execute_counter,
                        total: execute_total
                    }),
                );
                comptime!(execute_counter += 1);
            }
        }

        // Final matmul for n = n_iterations - 1.
        let last_idx = n_iterations - 1;
        let last_slot = last_idx % 2;

        #[unroll]
        for m_iter in 0..m_iterations {
            let accumulator = acc.tiles.index_mut(m_iter * n_iterations + last_idx);
            accumulator.mma(&a_fragment[m_iter], b_fragments.index(last_slot));

            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::TileMatmulCompleted {
                    current: execute_counter,
                    total: execute_total
                }),
            );
            comptime!(execute_counter += 1);
        }
    }

    assert!(a_load_counter == a_load_total);
    assert!(b_load_counter == b_load_total);
    assert!(execute_counter == execute_total);
    SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
}
