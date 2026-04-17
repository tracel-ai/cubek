use crate::{
    components::{
        global::{self, PlaneFlowPartitionRule, WriteEventListener},
        stage::{
            NoEvent, Stage, StageConfig, StageEventListener, StageMatmul,
            matmul::{
                partition::{Accumulators, PartitionMatmul, RhsTile, SharedPartitionMatmulConfig},
                plane_partitioned::PlanePartitionedStageConfig,
                scheduler::PartitionScheduler,
                unit_partitioned::UnitPartitionedStageConfig,
            },
        },
        tile::{TileConfig, TileMatmul, Tilex},
    },
    definition::{MatmulTypes, MatrixTypes},
};
use core::marker::PhantomData;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::stage::StageMemoryConfig;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait StagePartitioner: Send + Sync + 'static {
    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates(
        #[comptime] role_rule_config: PlaneFlowPartitionRule,
        #[comptime] plane_dim: u32,
        #[comptime] num_partitions_col: u32,
    ) -> Coords2d;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PartitionMatmulConfig<TC: TileConfig> {
    Unit(UnitPartitionedStageConfig<TC>),
    Plane(PlanePartitionedStageConfig<TC>),
}

impl<TC: TileConfig> PartitionMatmulConfig<TC> {
    pub fn shared(&self) -> SharedPartitionMatmulConfig<TC> {
        match self {
            PartitionMatmulConfig::Unit(unit_partitioned_stage_config) => {
                unit_partitioned_stage_config.shared
            }
            PartitionMatmulConfig::Plane(plane_partitioned_stage_config) => {
                plane_partitioned_stage_config.shared
            }
        }
    }
}

impl<TC: TileConfig> StageConfig for PartitionMatmulConfig<TC> {
    type TileConfig = TC;

    fn elements_in_stage_m(&self) -> u32 {
        self.shared().stage_size.m()
            * self.shared().partition_size.m()
            * self.shared().tile_config.elements_in_tile_m()
    }

    fn elements_in_stage_n(&self) -> u32 {
        self.shared().stage_size.n()
            * self.shared().partition_size.n()
            * self.shared().tile_config.elements_in_tile_n()
    }

    fn elements_in_stage_k(&self) -> u32 {
        self.shared().stage_size.k()
            * self.shared().partition_size.k()
            * self.shared().tile_config.elements_in_tile_k()
    }

    fn num_main_flow_planes(&self) -> u32 {
        self.shared().plane_flow_config.main_flow_count()
    }

    fn plane_dim(&self) -> u32 {
        self.shared().plane_dim
    }

    fn plane_flow_config(&self) -> global::PlaneFlowConfig {
        self.shared().plane_flow_config
    }

    fn tiles_in_partition_mn(&self) -> u32 {
        let partition_size = self.shared().partition_size;
        partition_size.m() * partition_size.n()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.shared().tile_config.elements_in_tile_k()
    }

    fn lhs_smem_config(&self) -> StageMemoryConfig {
        self.shared().lhs_smem_config
    }

    fn rhs_smem_config(&self) -> StageMemoryConfig {
        self.shared().rhs_smem_config
    }

    fn acc_smem_config(&self) -> StageMemoryConfig {
        self.shared().acc_smem_config
    }

    fn out_smem_config(&self) -> StageMemoryConfig {
        self.shared().out_smem_config
    }
}

/// Stage Matmul implementation that splits its stage across partitions, one per compute primitive.
///
/// Its results are written in a temporary shared memory to correct the layout before storing to global memory.
pub struct PartitionedStageMatmul<
    MP: MatmulTypes,
    TM: TileMatmul<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Lhs as MatrixTypes>::RegisterSize,
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::RegisterSize,
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
        >,
    StageLhs: Stage<
            <<MP as MatmulTypes>::Lhs as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Lhs as MatrixTypes>::StageSize,
            ReadOnly,
        >,
    StageRhs: Stage<
            <<MP as MatmulTypes>::Rhs as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Rhs as MatrixTypes>::StageSize,
            ReadOnly,
        >,
    StageAcc: Stage<
            <<MP as MatmulTypes>::Acc as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Acc as MatrixTypes>::StageSize,
            ReadOnly,
        >,
    StageOut: Stage<
            <<MP as MatmulTypes>::Acc as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Acc as MatrixTypes>::StageSize,
            ReadWrite,
        >,
    SP: StagePartitioner,
> {
    #[allow(clippy::type_complexity)]
    _phantom: PhantomData<(MP, TM, StageLhs, StageRhs, StageAcc, StageOut, SP)>,
}

#[cube]
impl<MP, TM, StageLhs, StageRhs, StageAcc, StageOut, SP> StageMatmul<MP>
    for PartitionedStageMatmul<MP, TM, StageLhs, StageRhs, StageAcc, StageOut, SP>
where
    MP: MatmulTypes,
    TM: TileMatmul<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Lhs as MatrixTypes>::RegisterSize,
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::RegisterSize,
            <MP::Acc as MatrixTypes>::Register,
            <MP::Acc as MatrixTypes>::RegisterSize,
        >,
    StageLhs: Stage<
            <<MP as MatmulTypes>::Lhs as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Lhs as MatrixTypes>::StageSize,
            ReadOnly,
        >,
    StageRhs: Stage<
            <<MP as MatmulTypes>::Rhs as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Rhs as MatrixTypes>::StageSize,
            ReadOnly,
        >,
    StageAcc: Stage<
            <<MP as MatmulTypes>::Acc as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Acc as MatrixTypes>::StageSize,
            ReadOnly,
        >,
    StageOut: Stage<
            <<MP as MatmulTypes>::Acc as MatrixTypes>::Stage,
            <<MP as MatmulTypes>::Acc as MatrixTypes>::StageSize,
            ReadWrite,
        >,
    SP: StagePartitioner,
{
    type Config = PartitionMatmulConfig<TM::Config>;

    type LhsStage = StageLhs;
    type RhsStage = StageRhs;
    type AccStage = StageAcc;
    type OutStage = StageOut;

    type Accumulators = Accumulators<MP, TM>;
    type LhsTile = Sequence<
        Tilex<
            <MP::Lhs as MatrixTypes>::Register,
            <MP::Lhs as MatrixTypes>::RegisterSize,
            ReadWrite,
        >,
    >;
    type RhsTile = RhsTile<
        Tilex<
            <MP::Rhs as MatrixTypes>::Register,
            <MP::Rhs as MatrixTypes>::RegisterSize,
            ReadWrite,
        >,
    >;

    fn execute(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        partition_scheduler: &PartitionScheduler,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_stage,
            rhs_stage,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            NoEvent::new(),
            partition_scheduler,
        )
    }

    fn execute_with_listener<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc>::execute_with_listener::<SEL>(
            lhs_stage,
            rhs_stage,
            lhs_fragment,
            rhs_fragments,
            acc,
            config.shared(),
            listener,
            partition_scheduler,
        );
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc>::init_tile_inputs(config.shared())
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc>::init_accumulator(config.shared())
    }

    fn load_accumulators(
        stage: &Self::AccStage,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
    ) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc>::load_accumulator(
            stage,
            acc,
            config.shared(),
        );
    }

    fn write_results<W: WriteEventListener>(
        acc: &mut Self::Accumulators,
        stage: &mut Self::OutStage,
        listener: &mut W,
        partition_scheduler: &PartitionScheduler,
        #[comptime] stage_config: Self::Config,
    ) {
        let m_iterations = stage_config.shared().partition_size.m() as usize;
        let n_iterations = stage_config.shared().partition_size.n() as usize;

        W::on_event(listener, global::WriteEvent::new_Begin());

        // Iterate over each tile in the partition
        #[unroll]
        for m_iter in 0..m_iterations {
            let m_load_iter = partition_scheduler.map_m(m_iter as u32);

            #[unroll]
            for n_iter in 0..n_iterations {
                let n_load_iter = partition_scheduler.map_n(n_iter as u32);

                let tile_accumulator = Accumulators::<MP, TM>::get_at_mut(
                    acc,
                    m_iter,
                    n_iter,
                    stage_config.shared().partition_size.n() as usize,
                );

                let tile_pos = (m_load_iter, n_load_iter);
                let mut tile = Self::OutStage::tile(stage, tile_pos);

                // Write the results for one tile. To save shared memory space, it reuses the same spot for
                // all tiles in the partition
                TM::write_results(
                    &mut tile,
                    tile_accumulator,
                    stage_config.shared().tile_config,
                );
                W::on_event(listener, global::WriteEvent::new_TileStored(tile_pos));
            }
        }

        W::on_event(listener, global::WriteEvent::new_Finish());
    }

    fn init_scheduler(#[comptime] config: Self::Config) -> PartitionScheduler {
        let (partition_row, partition_col) = SP::coordinates(
            config.shared().plane_flow_config.partition_rule,
            config.shared().plane_dim,
            config.shared().stage_size.n(),
        );

        PartitionScheduler::new(
            partition_row,
            partition_col,
            config.shared().partition_size,
            config.shared().partition_schedule_scheme,
        )
    }
}
