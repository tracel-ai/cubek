use std::marker::PhantomData;

use super::fragments::{Accumulators, RhsTile, RhsTileExpand};
use crate::{
    components::{
        global::PlaneFlowConfig,
        stage::{
            PartitionBuffering, Stage, StageEvent, StageEventListener,
            matmul::scheduler::PartitionScheduler,
        },
        tile_matmul::{
            Scope, TileMatmul, cmma_allocate_lhs, cmma_allocate_rhs, interleaved_allocate_lhs,
            interleaved_allocate_rhs, mma_allocate_lhs, mma_allocate_rhs, planevec_allocate_lhs,
            planevec_allocate_rhs, register_allocate_lhs, register_allocate_rhs,
        },
    },
    definition::{
        AccRE, Lhs, LhsRE, LhsRS, LhsSE, LhsSS, MatmulTypes, MatrixTypes, RhsRE, RhsRS, RhsSE,
        RhsSS, StageIdent,
    },
};
use crate::{
    components::{stage::PartitionSchedulerScheme, tile_matmul::Tile},
    definition::{Acc, Rhs},
};
use cubecl::prelude::*;
use cubek_std::{PartitionSize, StageSize, stage::StageMemoryConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedPartitionMatmulConfig {
    pub tile_config: TileMatmul,
    pub partition_size: PartitionSize,
    pub partition_buffering: PartitionBuffering,
    pub plane_flow_config: PlaneFlowConfig,
    pub plane_dim: u32,
    pub stage_size: StageSize,
    pub partition_schedule_scheme: PartitionSchedulerScheme,
    pub lhs_smem_config: StageMemoryConfig,
    pub rhs_smem_config: StageMemoryConfig,
    pub acc_smem_config: StageMemoryConfig,
    pub out_smem_config: StageMemoryConfig,
}

impl SharedPartitionMatmulConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tile_config: TileMatmul,
        partition_size: PartitionSize,
        partition_buffering: PartitionBuffering,
        plane_flow_config: PlaneFlowConfig,
        plane_dim: u32,
        stage_size: StageSize,
        partition_schedule_scheme: PartitionSchedulerScheme,
        lhs_smem_config: StageMemoryConfig,
        rhs_smem_config: StageMemoryConfig,
        acc_smem_config: StageMemoryConfig,
        out_smem_config: StageMemoryConfig,
    ) -> Self {
        Self {
            tile_config,
            partition_size,
            partition_buffering,
            plane_flow_config,
            plane_dim,
            stage_size,
            partition_schedule_scheme,
            lhs_smem_config,
            rhs_smem_config,
            acc_smem_config,
            out_smem_config,
        }
    }
}

type STy<T> = crate::definition::Stage<T>;
type SSz<T> = crate::definition::StageSize<T>;

/// Matmul for a whole partition, a region of the Stage Matmul
/// executed by a single compute primitive (unit or plane)
pub struct PartitionMatmul<
    MP: MatmulTypes,
    StageLhs: Stage<STy<Lhs<MP>>, SSz<Lhs<MP>>, ReadOnly>,
    StageRhs: Stage<STy<Rhs<MP>>, SSz<Rhs<MP>>, ReadOnly>,
    StageAcc: Stage<STy<Acc<MP>>, SSz<Acc<MP>>, ReadOnly>,
    Sc: Scope,
> {
    _phantom: PhantomData<(MP, StageLhs, StageRhs, StageAcc, Sc)>,
}

#[cube]
impl<MT, StageLhs, StageRhs, StageAcc, Sc> PartitionMatmul<MT, StageLhs, StageRhs, StageAcc, Sc>
where
    MT: MatmulTypes,
    StageLhs: Stage<STy<Lhs<MT>>, SSz<Lhs<MT>>, ReadOnly>,
    StageRhs: Stage<STy<Rhs<MT>>, SSz<Rhs<MT>>, ReadOnly>,
    StageAcc: Stage<STy<Acc<MT>>, SSz<Acc<MT>>, ReadOnly>,
    Sc: Scope,
{
    #[allow(clippy::too_many_arguments)]
    /// Execute all Tile Matmuls inside the partition
    /// Can be with single or double buffering
    pub fn execute_with_listener<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Sequence<
            Tile<
                <MT::Lhs as MatrixTypes>::Register,
                <MT::Lhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        >,
        rhs_fragments: &mut RhsTile<
            Tile<
                <MT::Rhs as MatrixTypes>::Register,
                <MT::Rhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        >,
        acc: &mut Accumulators<MT, Sc>,
        #[comptime] shared_config: SharedPartitionMatmulConfig,
        listener: SEL,
        partition_iterator: &PartitionScheduler,
    ) {
        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<SEL>(
                lhs_stage,
                rhs_stage,
                lhs_fragment,
                rhs_fragment,
                acc,
                shared_config,
                listener,
                partition_iterator,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<SEL>(
                lhs_stage,
                rhs_stage,
                lhs_fragment,
                rhs_fragments,
                acc,
                shared_config,
                listener,
                partition_iterator,
            ),
        }
    }

    /// Initialize Lhs and Rhs inputs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to load inputs before execution.
    pub fn init_tile_inputs(
        #[comptime] shared_config: SharedPartitionMatmulConfig,
    ) -> (
        Sequence<
            Tile<
                <MT::Lhs as MatrixTypes>::Register,
                <MT::Lhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        >,
        RhsTile<
            Tile<
                <MT::Rhs as MatrixTypes>::Register,
                <MT::Rhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        >,
    ) {
        let mut lhs = Sequence::new();

        #[unroll]
        for _ in 0..shared_config.partition_size.m() {
            lhs.push(allocate_lhs::<MT, Sc>(
                shared_config.lhs_smem_config.matrix_layout,
                shared_config.tile_config,
            ));
        }

        let rhs = match shared_config.partition_buffering {
            PartitionBuffering::Single => RhsTile::new_Single(allocate_rhs::<MT, Sc>(
                shared_config.rhs_smem_config.matrix_layout,
                shared_config.tile_config,
            )),
            PartitionBuffering::Double => RhsTile::new_Double((
                allocate_rhs::<MT, Sc>(
                    shared_config.rhs_smem_config.matrix_layout,
                    shared_config.tile_config,
                ),
                allocate_rhs::<MT, Sc>(
                    shared_config.rhs_smem_config.matrix_layout,
                    shared_config.tile_config,
                ),
            )),
        };

        (lhs, rhs)
    }

    /// Initialize accumulators
    ///     
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call `load_accumulator` prior to execute_with_listener.
    pub fn init_accumulator(
        #[comptime] shared_config: SharedPartitionMatmulConfig,
    ) -> Accumulators<MT, Sc> {
        Accumulators::<MT, Sc>::new(
            shared_config.partition_size,
            shared_config.out_smem_config.matrix_layout,
            shared_config.tile_config,
        )
    }

    /// Fill accumulators through a stage
    pub fn load_accumulator(
        stage: &StageAcc,
        acc: &mut Accumulators<MT, Sc>,
        partition_scheduler: &PartitionScheduler,
        #[comptime] shared_config: SharedPartitionMatmulConfig,
    ) {
        acc.load::<StageAcc>(
            stage,
            partition_scheduler,
            shared_config.partition_size.m() as usize,
            shared_config.partition_size.n() as usize,
        );
    }

    /// Execute partition matmul with a single buffer for rhs.
    ///
    /// This function can call functions at various events through the listener.
    #[allow(clippy::too_many_arguments)]
    fn execute_single_buffer<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Sequence<
            Tile<
                <MT::Lhs as MatrixTypes>::Register,
                <MT::Lhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        >,
        rhs_fragment: &mut Tile<
            <MT::Rhs as MatrixTypes>::Register,
            <MT::Rhs as MatrixTypes>::RegisterSize,
            Sc,
            ReadWrite,
        >,
        acc: &mut Accumulators<MT, Sc>,
        #[comptime] shared_config: SharedPartitionMatmulConfig,
        mut listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let m_iterations = shared_config.partition_size.m() as usize;
        let n_iterations = shared_config.partition_size.n() as usize;
        let k_iterations = shared_config.partition_size.k() as usize;

        let mut lhs_load_counter = 0.comptime();
        let mut rhs_load_counter = 0.comptime();
        let mut execute_counter = 0.comptime();
        let lhs_load_total = (m_iterations * k_iterations) as u32;
        let rhs_load_total = (n_iterations * k_iterations) as u32;
        let execute_total = (m_iterations * n_iterations * k_iterations) as u32;

        #[unroll]
        for k_iter in 0..k_iterations {
            let k_load_iter = partition_scheduler.map_k(k_iter as u32);

            #[unroll]
            for m_iter in 0..m_iterations {
                let m_load_iter = partition_scheduler.map_m(m_iter as u32);

                let tile_lhs = StageLhs::tile::<Sc>(lhs_stage, (m_load_iter, k_load_iter));

                lhs_fragment
                    .index_mut(m_iter)
                    .copy_from::<LhsSE<MT>, LhsSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
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
                let n_load_iter = partition_scheduler.map_n(n_iter as u32);

                let rhs_tile_next = StageRhs::tile::<Sc>(rhs_stage, (k_load_iter, n_load_iter));
                rhs_fragment
                    .copy_from::<RhsSE<MT>, RhsSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
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
                    let accumulator =
                        Accumulators::<MT, Sc>::get_at_mut(acc, m_iter, n_iter, n_iterations);
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

    #[allow(clippy::too_many_arguments)]
    /// Execute partition matmul with a double buffering for rhs.
    ///
    /// This function can call functions at various events through the listener.
    fn execute_double_buffer<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Sequence<
            Tile<
                <MT::Lhs as MatrixTypes>::Register,
                <MT::Lhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        >,
        rhs_fragments: &mut (
            Tile<
                <MT::Rhs as MatrixTypes>::Register,
                <MT::Rhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
            Tile<
                <MT::Rhs as MatrixTypes>::Register,
                <MT::Rhs as MatrixTypes>::RegisterSize,
                Sc,
                ReadWrite,
            >,
        ),
        acc: &mut Accumulators<MT, Sc>,
        #[comptime] shared_config: SharedPartitionMatmulConfig,
        mut listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let m_iterations = shared_config.partition_size.m() as usize;
        let n_iterations = shared_config.partition_size.n() as usize;
        let k_iterations = shared_config.partition_size.k() as usize;

        let mut lhs_load_counter = 0.comptime();
        let mut rhs_load_counter = 0.comptime();
        let mut execute_counter = 0.comptime();
        let lhs_load_total = (m_iterations * k_iterations) as u32;
        let rhs_load_total = (n_iterations * k_iterations) as u32;
        let execute_total = (m_iterations * n_iterations * k_iterations) as u32;

        #[unroll]
        for k_iter in 0..k_iterations {
            let k_load_iter = partition_scheduler.map_k(k_iter as u32);

            #[unroll]
            for m_iter in 0..m_iterations {
                let m_load_iter = partition_scheduler.map_m(m_iter as u32);

                let tile_lhs = StageLhs::tile::<Sc>(lhs_stage, (m_load_iter, k_load_iter));

                lhs_fragment
                    .index_mut(m_iter)
                    .copy_from::<LhsSE<MT>, LhsSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
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

            let mut n_iter = 0usize.comptime();
            let n_load_iter = partition_scheduler.map_n(n_iter as u32);

            let rhs_tile_first = StageRhs::tile::<Sc>(rhs_stage, (k_load_iter, n_load_iter));
            rhs_fragments
                .0
                .copy_from::<RhsSE<MT>, RhsSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
                    &rhs_tile_first,
                    StageIdent::Rhs,
                );

            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: rhs_load_counter,
                    total: rhs_load_total
                }),
            );
            comptime!(rhs_load_counter += 1);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 1..n_iterations {
                let (current, next) = if comptime! {n_iter.is_multiple_of(2)} {
                    (&mut rhs_fragments.0, &mut rhs_fragments.1)
                } else {
                    (&mut rhs_fragments.1, &mut rhs_fragments.0)
                };

                let n_load_iter = partition_scheduler.map_n(comptime![n_iter as u32 + 1]);
                let rhs_tile_next = StageRhs::tile::<Sc>(rhs_stage, (k_load_iter, n_load_iter));
                next.copy_from::<RhsSE<MT>, RhsSS<MT>, LhsRE<MT>, RhsRE<MT>, AccRE<MT>, ReadOnly>(
                    &rhs_tile_next,
                    StageIdent::Rhs,
                );

                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }),
                );
                comptime!(rhs_load_counter += 1);

                #[unroll]
                for m_iter in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MT, Sc>::get_at_mut(acc, m_iter, n_iter, n_iterations);
                    accumulator.mma(&lhs_fragment[m_iter], current);

                    SEL::on_event(
                        &mut listener,
                        comptime!(StageEvent::TileMatmulCompleted {
                            current: execute_counter,
                            total: execute_total
                        }),
                    );
                    comptime!(execute_counter += 1);
                }

                comptime![n_iter += 1];
            }

            let last = if comptime! {n_iter.is_multiple_of(2)} {
                &mut rhs_fragments.0
            } else {
                &mut rhs_fragments.1
            };

            #[unroll]
            for m_iter in 0..m_iterations {
                let accumulator =
                    Accumulators::<MT, Sc>::get_at_mut(acc, m_iter, n_iter, n_iterations);
                accumulator.mma(&lhs_fragment[m_iter], last);

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

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }
}

#[cube]
fn allocate_lhs<MT: MatmulTypes, Sc: Scope>(
    #[comptime] layout: cubek_std::MatrixLayout,
    #[comptime] config: TileMatmul,
) -> Tile<LhsRE<MT>, LhsRS<MT>, Sc, ReadWrite> {
    match config {
        TileMatmul::Cmma(c) => cmma_allocate_lhs::<LhsRE<MT>, LhsRS<MT>, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => {
            mma_allocate_lhs::<LhsRE<MT>, LhsRS<MT>, RhsRE<MT>, AccRE<MT>, Sc>(layout, c)
        }
        TileMatmul::Register(c) => register_allocate_lhs::<LhsRE<MT>, LhsRS<MT>, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_lhs::<LhsRE<MT>, LhsRS<MT>, Sc>(layout, c),
        TileMatmul::Interleaved(c) => {
            interleaved_allocate_lhs::<LhsRE<MT>, LhsRS<MT>, Sc>(layout, c)
        }
    }
}

#[cube]
fn allocate_rhs<MT: MatmulTypes, Sc: Scope>(
    #[comptime] layout: cubek_std::MatrixLayout,
    #[comptime] config: TileMatmul,
) -> Tile<RhsRE<MT>, RhsRS<MT>, Sc, ReadWrite> {
    match config {
        TileMatmul::Cmma(c) => cmma_allocate_rhs::<RhsRE<MT>, RhsRS<MT>, Sc>(layout, c.tile_size),
        TileMatmul::Mma(c) => {
            mma_allocate_rhs::<RhsRE<MT>, RhsRS<MT>, LhsRE<MT>, AccRE<MT>, Sc>(layout, c)
        }
        TileMatmul::Register(c) => register_allocate_rhs::<RhsRE<MT>, RhsRS<MT>, Sc>(layout, c),
        TileMatmul::PlaneVec(c) => planevec_allocate_rhs::<RhsRE<MT>, RhsRS<MT>, Sc>(layout, c),
        TileMatmul::Interleaved(c) => {
            interleaved_allocate_rhs::<RhsRE<MT>, RhsRS<MT>, Sc>(layout, c)
        }
    }
}
