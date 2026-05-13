use crate::components::stage::{
    self, NoEvent, PartitionScheduler,
    matmul::{
        partition::{Accumulators, PartitionMatmul, RhsTile},
        partitioned_matmul::StagePartitioner,
    },
    stage_matmul::{StageMatmul, write_partition_to_stage},
};
use crate::{
    components::global::GlobalReaderConfig,
    components::global::PlaneFlowPartition,
    components::global::Specializer,
    components::global::SpecializerKind,
    components::global::multi_stage::DoubleBufferingEventListener,
    components::global::multi_stage::JobExecutor,
    components::global::read::StageBuffer,
    components::global::{GlobalConfig, GlobalWriter},
    components::global::{LoadingSides, read::SyncStrategy},
    definition::{Acc, Lhs, MatmulTypes, MatrixTypes, Rhs, Stage},
};
use cubecl::prelude::*;
use cubek_std::tile::Tile;

#[cube]
/// Read the first stage for both Lhs and Rhs
///
/// If there is specialization, will add a runtime if to determine the role of the plane
pub fn read_first<S: SyncStrategy, LJ: JobExecutor<S>, RJ: JobExecutor<S>>(
    lhs_global_reader: &mut LJ,
    rhs_global_reader: &mut RJ,
    barrier: &mut S::Barrier,
    specializer: &Specializer,
    #[comptime] stage_to_load: StageBuffer,
    #[comptime] lhs_config: GlobalReaderConfig,
    #[comptime] rhs_config: GlobalReaderConfig,
) {
    match specializer.kind.comptime() {
        SpecializerKind::Specialized {
            main_flow_loading_side,
            load_only_loading_side,
            role_rule_config,
        } => {
            let rule = PlaneFlowPartition::new(role_rule_config);
            if !rule.is_load_plane() {
                if main_flow_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_global_reader, barrier, stage_to_load, lhs_config);
                }
                if main_flow_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_global_reader, barrier, stage_to_load, rhs_config);
                }
            } else {
                if load_only_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_global_reader, barrier, stage_to_load, lhs_config);
                }
                if load_only_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_global_reader, barrier, stage_to_load, rhs_config);
                }
            }
        }
        SpecializerKind::NotSpecialized => {
            LJ::execute_whole_job(lhs_global_reader, barrier, stage_to_load, lhs_config);
            RJ::execute_whole_job(rhs_global_reader, barrier, stage_to_load, rhs_config);
        }
    };
}

#[cube]
/// Execute on the current stage while loading the next stage
///
/// If there is specialization, will add a runtime if to determine the role of the plane
#[allow(clippy::too_many_arguments)]
pub fn execute_current_and_read_next<
    MP: MatmulTypes,
    SP: StagePartitioner,
    LhsStage: stage::Stage<Stage<Lhs<MP>>, ReadOnly>,
    RhsStage: stage::Stage<Stage<Rhs<MP>>, ReadOnly>,
    AccStage: stage::Stage<Stage<Acc<MP>>, ReadOnly>,
    S: SyncStrategy,
    LJ: JobExecutor<S>,
    RJ: JobExecutor<S>,
    G: GlobalConfig<StageConfig = StageMatmul>,
>(
    lhs_stage: &LhsStage,
    rhs_stage: &RhsStage,
    lhs_tile: &mut Sequence<Tile<<MP::Lhs as MatrixTypes>::Register, SP::Scope, ReadWrite>>,
    rhs_tile: &mut RhsTile<Tile<<MP::Rhs as MatrixTypes>::Register, SP::Scope, ReadWrite>>,
    acc: &mut Accumulators<MP, SP::Scope>,
    lhs_global_reader: &mut LJ,
    rhs_global_reader: &mut RJ,
    barrier: &mut S::Barrier,
    specializer: &Specializer,
    partition_scheduler: &PartitionScheduler,
    #[comptime] stage_to_load: StageBuffer,
    #[comptime] config: G,
) {
    match specializer.kind.comptime() {
        SpecializerKind::Specialized {
            main_flow_loading_side,
            load_only_loading_side,
            role_rule_config,
        } => {
            let rule = PlaneFlowPartition::new(role_rule_config);
            if !rule.is_load_plane() {
                PartitionMatmul::<MP, LhsStage, RhsStage, AccStage, SP::Scope>::execute_with_listener::<
                    DoubleBufferingEventListener<S, LJ, RJ, G>,
                >(
                    lhs_stage,
                    rhs_stage,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    config.stage_config().shared(),
                    DoubleBufferingEventListener::new(
                        stage_to_load,
                        lhs_global_reader,
                        rhs_global_reader,
                        barrier,
                        config,
                        main_flow_loading_side,
                    ),
                    partition_scheduler,
                );
            } else {
                if load_only_loading_side.includes_lhs() {
                    LJ::execute_whole_job(
                        lhs_global_reader,
                        barrier,
                        stage_to_load,
                        config.lhs_reader_config(),
                    );
                }
                if load_only_loading_side.includes_rhs() {
                    RJ::execute_whole_job(
                        rhs_global_reader,
                        barrier,
                        stage_to_load,
                        config.rhs_reader_config(),
                    );
                }
            }
        }
        SpecializerKind::NotSpecialized => {
            PartitionMatmul::<MP, LhsStage, RhsStage, AccStage, SP::Scope>::execute_with_listener::<
                DoubleBufferingEventListener<S, LJ, RJ, G>,
            >(
                lhs_stage,
                rhs_stage,
                lhs_tile,
                rhs_tile,
                acc,
                config.stage_config().shared(),
                DoubleBufferingEventListener::new(
                    stage_to_load,
                    lhs_global_reader,
                    rhs_global_reader,
                    barrier,
                    config,
                    LoadingSides::Both,
                ),
                partition_scheduler,
            );
        }
    };
}

#[cube]
/// Execute on the last stage, then write results
///
/// If there is specialization, will add a runtime if to determine the role of the plane
#[allow(clippy::too_many_arguments)]
pub fn execute_last_and_write_results<
    MP: MatmulTypes,
    GW: GlobalWriter<MP::Acc>,
    SP: StagePartitioner,
    LhsStage: stage::Stage<Stage<Lhs<MP>>, ReadOnly>,
    RhsStage: stage::Stage<Stage<Rhs<MP>>, ReadOnly>,
    AccStage: stage::Stage<Stage<Acc<MP>>, ReadOnly>,
    G: GlobalConfig<StageConfig = StageMatmul>,
>(
    lhs_stage: &LhsStage,
    rhs_stage: &RhsStage,
    lhs_tile: &mut Sequence<Tile<<MP::Lhs as MatrixTypes>::Register, SP::Scope, ReadWrite>>,
    rhs_tile: &mut RhsTile<Tile<<MP::Rhs as MatrixTypes>::Register, SP::Scope, ReadWrite>>,
    acc: &mut Accumulators<MP, SP::Scope>,
    out_writer: &mut GW,
    specializer: &Specializer,
    partition_scheduler: &PartitionScheduler,
    #[comptime] config: G,
) {
    let mut out_stage = GW::stage(out_writer);

    match specializer.kind.comptime() {
        SpecializerKind::Specialized {
            main_flow_loading_side: _,
            load_only_loading_side: _,
            role_rule_config,
        } => {
            let rule = PlaneFlowPartition::new(role_rule_config);
            if !rule.is_load_plane() {
                PartitionMatmul::<MP, LhsStage, RhsStage, AccStage, SP::Scope>::execute_with_listener::<
                    NoEvent,
                >(
                    lhs_stage,
                    rhs_stage,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    config.stage_config().shared(),
                    NoEvent::new(),
                    partition_scheduler,
                );

                write_partition_to_stage::<MP, SP::Scope, GW::Stage, GW>(
                    acc,
                    &mut out_stage,
                    out_writer,
                    partition_scheduler,
                    config.stage_config().shared(),
                );
            }
        }
        SpecializerKind::NotSpecialized => {
            PartitionMatmul::<MP, LhsStage, RhsStage, AccStage, SP::Scope>::execute_with_listener::<
                NoEvent,
            >(
                lhs_stage,
                rhs_stage,
                lhs_tile,
                rhs_tile,
                acc,
                config.stage_config().shared(),
                NoEvent::new(),
                partition_scheduler,
            );

            write_partition_to_stage::<MP, SP::Scope, GW::Stage, GW>(
                acc,
                &mut out_stage,
                out_writer,
                partition_scheduler,
                config.stage_config().shared(),
            );
        }
    }
}
