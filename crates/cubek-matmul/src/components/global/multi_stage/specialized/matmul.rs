use crate::components::global::read::{PartialStageGlobalReader, StageBuffer};
use crate::components::global::{
    GlobalWriter,
    read::{FullLoaderStage, FullLoadingStrategy, SyncStrategy},
};
use crate::{
    components::global::read::{FullStageGlobalReader, PartialLoaderStage},
    definition::Stage,
};
use crate::{
    components::global::{GlobalMatmul, SharedGlobalMatmulConfig},
    components::global::{PlaneFlowPartition, read::AsyncPartialLoadingStrategy},
    components::stage::{
        {StagePartitioner, partition_coordinates},
        {init_a_fragment, init_accumulator, init_b_fragments},
    },
    definition::*,
    launch::RuntimeConfig,
};

use cubecl::{
    prelude::barrier::Barrier,
    prelude::*,
    std::tensor::{View, layout::Coords2d},
};
use cubek_std::tile::{
    NoEvent, PartitionScheduler, Tile, load_partition_from_stage, write_partition_to_stage,
};
use std::marker::PhantomData;

// Per-flow Stage type aliases — keep call sites readable.
type LhsStageFor<MP, RC, L> = PartialLoaderStage<RC, L, Stage<Lhs<MP>>, StageSize<Lhs<MP>>>;
type RhsStageFor<MP, RC, L> = PartialLoaderStage<RC, L, Stage<Rhs<MP>>, StageSize<Rhs<MP>>>;
type AccStageFor<MP, RC, AL> =
    ComptimeOption<FullLoaderStage<RC, AL, Stage<Acc<MP>>, StageSize<Acc<MP>>>>;

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on stage A,
/// they trigger a computation event from tensor cores on stage B. Then stages are switched.
/// Specializes planes to either read or compute planes.
/// Hardcoded for TMA right now
pub struct SpecializedMatmul<
    MP: MatmulTypes,
    SP: StagePartitioner,
    RC: RuntimeConfig,
    L: AsyncPartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
> {
    _ms: PhantomData<MP>,
    _sp: PhantomData<SP>,
    _rc: PhantomData<RC>,
    _loading: PhantomData<L>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

#[cube]
impl<MP: MatmulTypes, SP, RC, L, AL, GW> GlobalMatmul<RC, MP>
    for SpecializedMatmul<MP, SP, RC, L, AL, GW>
where
    SP: StagePartitioner,
    RC: RuntimeConfig,
    L: AsyncPartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig;

    type LhsGlobalReader = PartialStageGlobalReader<
        <MP::Lhs as MatrixTypes>::Global,
        <MP::Lhs as MatrixTypes>::GlobalSize,
        <MP::Lhs as MatrixTypes>::Stage,
        <MP::Lhs as MatrixTypes>::StageSize,
        RC,
        L,
    >;
    type RhsGlobalReader = PartialStageGlobalReader<
        <MP::Rhs as MatrixTypes>::Global,
        <MP::Rhs as MatrixTypes>::GlobalSize,
        <MP::Rhs as MatrixTypes>::Stage,
        <MP::Rhs as MatrixTypes>::StageSize,
        RC,
        L,
    >;
    type AccGlobalReader = ComptimeOption<
        FullStageGlobalReader<
            <MP::Acc as MatrixTypes>::Global,
            <MP::Acc as MatrixTypes>::GlobalSize,
            <MP::Acc as MatrixTypes>::Stage,
            <MP::Acc as MatrixTypes>::StageSize,
            RC,
            AL,
        >,
    >;

    type GlobalWriter = GW;
    type Accumulators = Tile<AccRE<MP>, SP::Scope>;

    fn execute(
        mut lhs_reader: Self::LhsGlobalReader,
        mut rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        mut out_writer: Self::GlobalWriter,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let device_props = comptime::device_properties();
        if let Err(e) = comptime!(L::validate_with_config(
            &device_props,
            &config.lhs_reader_config
        )) {
            push_validation_error(e.to_string());
            comptime!(return);
        }
        if let Err(e) = comptime!(L::validate_with_config(
            &device_props,
            &config.rhs_reader_config
        )) {
            push_validation_error(e.to_string());
            comptime!(return);
        }

        let stage_step = config.stage_config.elements_in_stage_k();

        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = range.div_ceil(stage_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = num_stage_matmuls / 2;

        let stage_shared = config.stage_config.shared();

        let (partition_row, partition_col) = partition_coordinates::<SP>(
            stage_shared.plane_flow_config.partition_rule,
            stage_shared.plane_dim,
            stage_shared.stage_size.n(),
        );
        let partition_scheduler = PartitionScheduler::new(
            partition_row,
            partition_col,
            stage_shared.partition_size,
            stage_shared.partition_schedule_scheme,
        );

        let lhs_stage_a = lhs_reader.stage(StageBuffer::A);
        let lhs_stage_b = lhs_reader.stage(StageBuffer::B);
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);
        let lhs_stage_a_tile = <LhsStageFor<MP, RC, L> as crate::components::stage::Stage<
            Stage<Lhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&lhs_stage_a);
        let lhs_stage_b_tile = <LhsStageFor<MP, RC, L> as crate::components::stage::Stage<
            Stage<Lhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&lhs_stage_b);
        let rhs_stage_a_tile = <RhsStageFor<MP, RC, L> as crate::components::stage::Stage<
            Stage<Rhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&rhs_stage_a);
        let rhs_stage_b_tile = <RhsStageFor<MP, RC, L> as crate::components::stage::Stage<
            Stage<Rhs<MP>>,
        >>::as_stage_tile::<SP::Scope>(&rhs_stage_b);

        let compute_units = config.plane_flow_config().counts.main_flow * config.plane_dim();

        let role_rule = PlaneFlowPartition::new(config.plane_flow_config().partition_rule);

        let mut acc_barrier = AL::SyncStrategy::create_barrier();
        let acc_stage = acc_reader.map(|mut reader| {
            reader.load_stage(&mut acc_barrier, config.acc_reader_config);
            sync_cube();
            reader.stage()
        });

        // Barrier for writing out
        let barrier_done = Barrier::shared_uninit();

        // Barriers for releasing smem after compute
        let barrier_empty_a = Barrier::shared_uninit();
        let barrier_empty_b = Barrier::shared_uninit();

        // Barriers for marking smem as loaded
        let mut barrier_full_a = Barrier::shared_uninit();
        let mut barrier_full_b = Barrier::shared_uninit();

        if role_rule.elect_load_leader() {
            barrier_done.init_manual(compute_units);

            barrier_empty_a.init_manual(compute_units);
            barrier_empty_b.init_manual(compute_units);

            barrier_full_a.init_manual(L::arrival_count(config));
            barrier_full_b.init_manual(L::arrival_count(config));

            L::barrier_post_init();
        }
        sync_cube();

        let mut phase = 0;

        if L::is_elected(config) {
            for _ in 0..num_loops {
                barrier_empty_a.wait_parity(phase ^ 1);
                lhs_reader.load_stage(
                    &mut barrier_full_a,
                    StageBuffer::A,
                    config.lhs_reader_config,
                );
                rhs_reader.load_stage(
                    &mut barrier_full_a,
                    StageBuffer::A,
                    config.rhs_reader_config,
                );
                L::arrive::<MP>(&mut barrier_full_a, config);

                barrier_empty_b.wait_parity(phase ^ 1);
                lhs_reader.load_stage(
                    &mut barrier_full_b,
                    StageBuffer::B,
                    config.lhs_reader_config,
                );
                rhs_reader.load_stage(
                    &mut barrier_full_b,
                    StageBuffer::B,
                    config.rhs_reader_config,
                );
                L::arrive::<MP>(&mut barrier_full_b, config);

                lhs_reader.advance_view();
                rhs_reader.advance_view();
                phase ^= 1;
            }
        } else if role_rule.is_compute_plane() {
            let mut lhs_tile = init_a_fragment::<MP, SP::Scope>(stage_shared);
            let mut rhs_tile = init_b_fragments::<MP, SP::Scope>(stage_shared);
            let mut acc = init_accumulator::<MP, SP::Scope>(stage_shared);

            load_partition_from_stage::<
                AccSE<MP>,
                AccSS<MP>,
                LhsRE<MP>,
                RhsRE<MP>,
                AccRE<MP>,
                SP::Scope,
                AccStageFor<MP, RC, AL>,
            >(
                &acc_stage,
                &mut acc,
                &partition_scheduler,
                stage_shared.partition_size.m(),
                stage_shared.partition_size.n(),
            );

            for _ in 0..num_loops {
                barrier_full_a.wait_parity(phase);
                acc.mma_partition::<
                    LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                    RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                    NoEvent,
                >(
                    &lhs_stage_a_tile,
                    &rhs_stage_a_tile,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    stage_shared.partition_size.k(),
                    NoEvent::new(),
                    &partition_scheduler,
                );
                barrier_empty_a.arrive();

                barrier_full_b.wait_parity(phase);
                acc.mma_partition::<
                    LhsSE<MP>, LhsSS<MP>, LhsRE<MP>,
                    RhsSE<MP>, RhsSS<MP>, RhsRE<MP>,
                    NoEvent,
                >(
                    &lhs_stage_b_tile,
                    &rhs_stage_b_tile,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    stage_shared.partition_size.k(),
                    NoEvent::new(),
                    &partition_scheduler,
                );
                barrier_empty_b.arrive();

                phase ^= 1;
            }
            barrier_done.arrive_and_wait();

            lhs_reader.free_stage();
            rhs_reader.free_stage();

            let mut out_stage = Self::GlobalWriter::stage(&out_writer);

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
                &mut acc,
                &mut out_stage,
                &mut out_writer,
                &partition_scheduler,
                stage_shared.partition_size.m(),
                stage_shared.partition_size.n(),
            );
        }
    }

    fn init_lhs_global_reader(
        lhs: View<LhsG<MP>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        // We always advance by 2 * k because stage B shares the same global memory state as stage A,
        // but it is implicitly offset by one stage's worth (k elements) when reading.
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::new(lhs, runtime_config, k_step, config.lhs_reader_config)
    }

    fn init_rhs_global_reader(
        rhs: View<RhsG<MP>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        // We always advance by 2 * k because stage B shares the same global memory state as stage A,
        // but it is implicitly offset by one stage's worth (k elements) when reading.
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::new(rhs, runtime_config, k_step, config.rhs_reader_config)
    }

    fn init_acc_global_reader(
        acc: ComptimeOption<View<AccG<MP>, Coords2d>>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader {
        acc.map(|view| {
            FullStageGlobalReader::new(view, runtime_config, 0, config.acc_reader_config)
        })
    }

    fn init_global_writer(
        out: View<AccG<MP>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        Self::GlobalWriter::init(out, config.writer_config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        init_accumulator::<MP, SP::Scope>(config.stage_config.shared())
    }
}
