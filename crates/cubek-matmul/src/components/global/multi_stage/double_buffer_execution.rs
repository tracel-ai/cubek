#![allow(clippy::type_complexity)]

use crate::components::stage::StagePartitioner;
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
    definition::{
        AccRE, AccSS, LhsRE, LhsSE, LhsSS, MatmulTypes, MatrixTypes, RhsRE, RhsSE, RhsSS,
    },
};
use cubecl::prelude::*;
use cubek_std::tile::{NoEvent, PartitionScheduler, Tile, write_partition_to_stage};

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
    S: SyncStrategy,
    LJ: JobExecutor<S>,
    RJ: JobExecutor<S>,
    G: GlobalConfig,
>(
    lhs_stage: &Tile<<MP::Lhs as MatrixTypes>::Stage, SP::Scope>,
    rhs_stage: &Tile<<MP::Rhs as MatrixTypes>::Stage, SP::Scope>,
    lhs_tile: &mut Sequence<Tile<<MP::Lhs as MatrixTypes>::Register, SP::Scope>>,
    rhs_tile: &mut Tile<<MP::Rhs as MatrixTypes>::Register, SP::Scope>,
    acc: &mut Tile<AccRE<MP>, SP::Scope>,
    lhs_global_reader: &mut LJ,
    rhs_global_reader: &mut RJ,
    barrier: &mut S::Barrier,
    specializer: &Specializer,
    partition_scheduler: &PartitionScheduler,
    #[comptime] stage_to_load: StageBuffer,
    #[comptime] config: G,
) {
    let partition_size_k = comptime!(config.stage_config().shared().partition_size.k());
    match specializer.kind.comptime() {
        SpecializerKind::Specialized {
            main_flow_loading_side,
            load_only_loading_side,
            role_rule_config,
        } => {
            let rule = PlaneFlowPartition::new(role_rule_config);
            if !rule.is_load_plane() {
                acc.mma_partition::<
                    LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                    RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                    DoubleBufferingEventListener<S, LJ, RJ, G>,
                >(
                    lhs_stage,
                    rhs_stage,
                    lhs_tile,
                    rhs_tile,
                    partition_size_k,
                    DoubleBufferingEventListener::new(
                        stage_to_load,
                        &*lhs_global_reader,
                        &*rhs_global_reader,
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
            acc.mma_partition::<
                LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                DoubleBufferingEventListener<S, LJ, RJ, G>,
            >(
                lhs_stage,
                rhs_stage,
                lhs_tile,
                rhs_tile,
                partition_size_k,
                DoubleBufferingEventListener::new(
                    stage_to_load,
                    &*lhs_global_reader,
                    &*rhs_global_reader,
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
    G: GlobalConfig,
>(
    lhs_stage: &Tile<<MP::Lhs as MatrixTypes>::Stage, SP::Scope>,
    rhs_stage: &Tile<<MP::Rhs as MatrixTypes>::Stage, SP::Scope>,
    lhs_tile: &mut Sequence<Tile<<MP::Lhs as MatrixTypes>::Register, SP::Scope>>,
    rhs_tile: &mut Tile<<MP::Rhs as MatrixTypes>::Register, SP::Scope>,
    acc: &mut Tile<AccRE<MP>, SP::Scope>,
    out_writer: &mut GW,
    specializer: &Specializer,
    partition_scheduler: &PartitionScheduler,
    #[comptime] config: G,
) {
    let partition_size_k = comptime!(config.stage_config().shared().partition_size.k());
    let mut out_stage = GW::stage(&*out_writer);

    match specializer.kind.comptime() {
        SpecializerKind::Specialized {
            main_flow_loading_side: _,
            load_only_loading_side: _,
            role_rule_config,
        } => {
            let rule = PlaneFlowPartition::new(role_rule_config);
            if !rule.is_load_plane() {
                acc.mma_partition::<
                    LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                    RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                    NoEvent,
                >(
                    lhs_stage,
                    rhs_stage,
                    lhs_tile,
                    rhs_tile,
                    partition_size_k,
                    NoEvent::new(),
                    partition_scheduler,
                );

                write_partition_to_stage::<
                    <MP::Acc as MatrixTypes>::Stage,
                    AccSS<MP>,
                    LhsRE<MP>,
                    RhsRE<MP>,
                    AccRE<MP>,
                    SP::Scope,
                    GW::Stage,
                    GW,
                >(
                    acc,
                    &mut out_stage,
                    out_writer,
                    partition_scheduler,
                    config.stage_config().shared().partition_size.m(),
                    config.stage_config().shared().partition_size.n(),
                );
            }
        }
        SpecializerKind::NotSpecialized => {
            acc.mma_partition::<
                LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                NoEvent,
            >(
                lhs_stage,
                rhs_stage,
                lhs_tile,
                rhs_tile,
                partition_size_k,
                NoEvent::new(),
                partition_scheduler,
            );

            write_partition_to_stage::<
                <MP::Acc as MatrixTypes>::Stage,
                AccSS<MP>,
                LhsRE<MP>,
                RhsRE<MP>,
                AccRE<MP>,
                SP::Scope,
                GW::Stage,
                GW,
            >(
                acc,
                &mut out_stage,
                out_writer,
                partition_scheduler,
                config.stage_config().shared().partition_size.m(),
                config.stage_config().shared().partition_size.n(),
            );
        }
    }
}
